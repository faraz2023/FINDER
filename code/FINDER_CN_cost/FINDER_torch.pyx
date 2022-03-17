
import torch
from torch import nn
import torch.optim as optim
import torch_sparse

import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import heapq
import scipy.linalg as linalg
import os
import pandas as pd
import os.path
from torch.autograd import Variable

from FINDER_net import FINDER_net



# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
#cdef int MAX_ITERATION = 1000000
cdef int MAX_ITERATION = 400001

cdef double LEARNING_RATE = 0.0001   #dai
cdef int MEMORY_SIZE = 500000
cdef double Alpha = 0.001 ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5
cdef int NUM_MIN = 30
cdef int NUM_MAX = 50
cdef int REG_HIDDEN = 32
cdef int M = 4  # how many edges selected each time for BA model
cdef int BATCH_SIZE = 64
cdef double initialization_stddev = 0.01  # 权重初始化的方差
cdef int n_valid = 200
cdef int aux_dim = 4
cdef int num_env = 1
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 2
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage



class FINDER:

    def __init__(self):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'barabasi_albert'#erdos_renyi, powerlaw, small-world
        self.training_type = 'degree'   #'random'
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        if(self.IsHuberloss):
            self.loss = nn.HuberLoss(delta=1.0)
        else:
            self.loss = nn.MSELoss()
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False


        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False
        ############----------------------------- variants of DQN(end) ------------------- ###################################

        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        # self.covered=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)

        print("CUDA:", torch.cuda.is_available())
        #torch.set_num_threads(16)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.FINDER_net = FINDER_net(device=self.device)
        self.FINDER_net_T = FINDER_net(device=self.device)

        #self.FINDER_net = self.FINDER_net.double()
        #self.FINDER_net_T = self.FINDER_net.double()

        self.FINDER_net.to(self.device)
        self.FINDER_net_T.to(self.device)

        self.FINDER_net_T.eval()

        self.optimizer = optim.Adam(self.FINDER_net.parameters(), lr=self.learning_rate)

        pytorch_total_params = sum(p.numel() for p in self.FINDER_net.parameters())
        print("Total number of FINDER_net parameters: {}".format(pytorch_total_params))





    def gen_graph(self, num_min, num_max):
        cdef int max_n = num_max
        cdef int min_n = num_min
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)

        if self.training_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0,1)
        elif self.training_type == 'degree':
            degree = nx.degree(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]/maxDegree

        elif self.training_type == 'degree_centrality':
            degree = nx.degree_centrality(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]
        nx.set_node_attributes(g, weights, 'weight')
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        cdef int i
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    #pass
    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        cdef double result_degree = 0.0
        cdef double result_betweeness = 0.0

        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            g_degree = g.copy()
            g_betweenness = g.copy()
            result_degree += self.HXA(g_degree,'HDA')
            result_betweeness += self.HXA(g_betweenness, 'HBA')
            self.InsertGraph(g, is_test=True)
        print ('Validation of HDA: %.16f'%(result_degree / n_valid))
        print ('Validation of HBA: %.16f'%(result_betweeness / n_valid))


    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step):
        cdef int num_env = len(self.env_list)
        cdef int n = 0
        cdef int i
        while n < num_seq:
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step)
                    g_sample= TrainSet.Sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= num_seq:
                break

            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)
    #pass
    def PlayGame(self,int n_traj, double eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)


    def SetupSparseT(self, sparse_dict):
        sparse_dict['index'] = Variable(sparse_dict['index']).to(self.device)
        sparse_dict['value'] = Variable(sparse_dict['value']).to(self.device)

        return sparse_dict

    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = Variable(torch.tensor(self.m_y).type(torch.FloatTensor)).to(self.device)
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)

        self.inputs['action_select'] = self.SetupSparseT(prepareBatchGraph.act_select)
        self.inputs['rep_global'] = self.SetupSparseT(prepareBatchGraph.rep_global)
        self.inputs['n2nsum_param'] = self.SetupSparseT(prepareBatchGraph.n2nsum_param)
        self.inputs['laplacian_param'] = self.SetupSparseT(prepareBatchGraph.laplacian_param)
        self.inputs['subgsum_param'] = self.SetupSparseT(prepareBatchGraph.subgsum_param)

        self.inputs['node_input'] = Variable(torch.tensor(prepareBatchGraph.node_feat).type(torch.FloatTensor)).to(self.device)
        self.inputs['aux_input'] = Variable(torch.tensor(prepareBatchGraph.aux_feat).type(torch.FloatTensor)).to(self.device)

    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = self.SetupSparseT(prepareBatchGraph.rep_global)

        self.inputs['n2nsum_param'] = self.SetupSparseT(prepareBatchGraph.n2nsum_param)

        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = self.SetupSparseT(prepareBatchGraph.subgsum_param)

        self.inputs['node_input'] = Variable(torch.tensor(prepareBatchGraph.node_feat).type(torch.FloatTensor)).to(self.device)
        self.inputs['aux_input'] = Variable(torch.tensor(prepareBatchGraph.aux_feat).type(torch.FloatTensor)).to(self.device)
        return prepareBatchGraph.idx_map_list

    def Predict(self,g_list,covered,isSnapSnot):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            if isSnapSnot:
                result = self.FINDER_net_T.test_forward(node_input=self.inputs['node_input'],\
                    subgsum_param=self.inputs['subgsum_param'], n2nsum_param=self.inputs['n2nsum_param'],\
                    rep_global=self.inputs['rep_global'], aux_input=self.inputs['aux_input'])
            else:
                result = self.FINDER_net.test_forward(node_input=self.inputs['node_input'],\
                    subgsum_param=self.inputs['subgsum_param'], n2nsum_param=self.inputs['n2nsum_param'],\
                    rep_global=self.inputs['rep_global'], aux_input=self.inputs['aux_input'])
            # TOFIX: line below used to be raw_output = result[0]. This is weird because results is supposed to be 
            # [node_cnt, 1] (Q-values per node). And indeed it resulted in an error! I have fixed it by the line below
            # look inito it later.
            raw_output = result[:,0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self,g_list,covered):
        result = self.Predict(g_list,covered,False)
        return result

    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result
    #pass
    def TakeSnapShot(self):
        self.FINDER_net_T.load_state_dict(self.FINDER_net.state_dict())

    def Fit(self):
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False
        cdef int i
        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)

        list_target = np.zeros([BATCH_SIZE, 1])

        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=GAMMA * list_pred[i]
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        '''
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param : self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.aux_input : np.array(self.inputs['aux_input']),
                                        self.ISWeights : np.mat(ISWeights).T,
                                        self.target : self.inputs['target']})
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        # print ('loss')
        # print (loss / len(g_list))
        return loss / len(g_list)
        '''
        return None

    def fit(self,g_list,covered,actions,list_target):
        cdef double loss_values = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            self.optimizer.zero_grad()

            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            q_pred, cur_message_layer = self.FINDER_net.train_forward(node_input=self.inputs['node_input'],\
                subgsum_param=self.inputs['subgsum_param'], n2nsum_param=self.inputs['n2nsum_param'],\
                action_select=self.inputs['action_select'], aux_input=self.inputs['aux_input'])

            loss = self.calc_loss(q_pred, cur_message_layer)
            loss.backward()
            self.optimizer.step()

            loss_values += loss.item()*bsize

        return loss_values / len(g_list)

    def calc_loss(self, q_pred, cur_message_layer) :
        ## first order reconstruction loss
        #OLD loss_recons = 2 * torch.trace(torch.matmul(torch.transpose(cur_message_layer,0,1),\
        #    torch.matmul(self.inputs['laplacian_param'], cur_message_layer)))
        loss_recons = 2 * torch.trace(torch.matmul(torch.transpose(cur_message_layer,0,1),\
            torch_sparse.spmm(self.inputs['laplacian_param']['index'], self.inputs['laplacian_param']['value'],\
            self.inputs['laplacian_param']['m'], self.inputs['laplacian_param']['n'],\
             cur_message_layer)))
        edge_num = torch.sum(self.inputs['n2nsum_param']['value'])
        #edge_num = torch.sum(self.inputs['n2nsum_param'])

        loss_recons = torch.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = torch.sum(torch.abs(self.inputs['target'] - q_pred), dim=1)    # for updating Sumtree
            if self.IsHuberloss:
                pass
                #loss_rl = self.loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                pass
                #loss_rl = torch.sum(self.ISWeights * self.loss(self.target, q_pred))
        else:
            if self.IsHuberloss:
                pass
                #loss_rl = self.loss(self.inputs['target'], q_pred)
            else:
                loss_rl = self.loss(self.inputs['target'], q_pred)

        loss = torch.add(loss_rl, loss_recons, alpha = Alpha)

        return loss

    def Train(self, skip_saved_iter=False):
        self.PrepareValidData()
        self.gen_new_graphs(NUM_MIN, NUM_MAX)
        cdef int i, iter, idx
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        cdef double eps_start = 1.0
        cdef double eps_end = 0.05
        cdef double eps_step = 10000.0
        cdef int loss = 0
        cdef double frac, start, end

        save_dir = './models/truncNormal-TORCH-Model_%s'%(self.g_type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)

        start_iter=0
        if(skip_saved_iter):
            if(os.path.isfile(VCFile)):
                f_read = open(VCFile)
                line_ctr = f_read.read().count("\n")
                f_read.close()
                start_iter = max(300 * (line_ctr-1), 0)
                start_model = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, start_iter)
                print(f'Found VCFile {VCFile}, choose start model: {start_model}')
                if(os.path.isfile(VCFile)):
                    self.LoadModel(start_model)
                    print(f'skipping iterations that are already done, starting at iter {start_iter}..')                    
                    # append instead of new write
                    f_out = open(VCFile, 'a')
                else:
                    print('failed to load starting model, start iteration from 0..')
                    start_iter=0
                    f_out = open(VCFile, 'w')                
        else:
            f_out = open(VCFile, 'w')

        for iter in range(MAX_ITERATION):
            start = time.perf_counter()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if( (iter and iter % 5000 == 0) or (iter==start_iter)):
                self.gen_new_graphs(NUM_MIN, NUM_MAX)

            ###########-----------------------normal training data setup(end) -----------------##############################

            ###########-----------------------stratified training data setup(start) -----------------##############################
            # if iter <= 200000:
            #     if iter and iter % 5000 == 0:
            #         self.gen_new_graphs(15, 20)
            # elif iter > 200000 and iter <= 400000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(30, 50)
            # elif iter > 400000 and iter <= 600000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(50, 100)
            # elif iter > 600000 and iter <= 800000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(100, 200)
            # elif iter > 800000 and iter <= 1000000:
            #     if iter % 5000 == 0:
            #         self.gen_new_graphs(400, 500)
            ###########-----------------------stratified training data setup(end) -----------------##############################
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 500 == 0:
                if(iter == 0 or iter == start_iter):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                # n_valid = 1
                test_start = time.time()
                for idx in range(n_valid):
                    frac += self.Test(idx)
                test_end = time.time()
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush()
                print('iter', iter, 'eps', eps, 'average size of vc: ', frac / n_valid)
                print ('testing 100 graphs time: %.8fs'%(test_end-test_start))
                N_end = time.perf_counter()
                print ('300 iterations total time: %.8fs'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                if(skip_saved_iter and iter==start_iter):
                    pass
                else:
                    if(iter % 10000 == 0):
                        self.SaveModel(model_path)
            if( (iter % UPDATE_TIME == 0) or (iter==start_iter)):
                self.TakeSnapShot()
            self.Fit()
        f_out.close()


    def findModel(self):
        VCFile = './models/ModelVC_%d_%d.csv'%(NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        best_model = './models/nrange_%d_%d_iter_%d.ckpt' % (NUM_MIN, NUM_MAX, best_model_iter)
        return best_model


    def Evaluate(self, data_test, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef int i
        result_list_score = []
        result_list_time = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g_path = '%s/'%data_test + 'g_%d'%i
            g = nx.read_gml(g_path, destringizer=int)
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(i)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
        self.ClearTestGraphs()
        # print ('\nvc mean: ', np.mean(result_list))
        # print ('vc std: ', np.std(result_list))
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std

    def EvaluateReinsert(self, data_test,save_dir, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        cdef int i
        f = open(data_test, 'rb')
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g = cp.load(f)
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSolReinsert(i)
            t2 = time.time()
            frac += val
            frac_time += (t2 - t1)
        self.ClearTestGraphs()
        print ('average size of vc: ', frac / n_test)
        print('average time: ', frac_time / n_test)
        return  frac / n_test, frac_time / n_test

    def CleanRealData(self, data_test):
        G = nx.read_weighted_edgelist(data_test)
        cc = sorted(nx.connected_components(G), key = len, reverse=True)
        lcc = cc[0]
        # remove all nodes not in largest component
        numrealnodes = 0
        node_map = {}
        for node in G.nodes():
            if node not in lcc:
                G.remove_node(node)
                continue
            node_map[node] = numrealnodes
            numrealnodes += 1
        # re-create the largest component with nodes indexed from 0 sequentially
        g = nx.Graph()
        for edge in G.edges_iter(data=True):
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            # w = edge[2]['weight']
            # g.add_edge(src_idx,dst_idx,weight=w)
            g.add_edge(src_idx,dst_idx)
        nx.write_edgelist(g, './Real_data/real1.txt')


    def EvaluateRealData(self, model_file, data_test, save_dir, stepRatio=0.0025):  #测试真实数据
        cdef double solution_time = 0.0
        test_name = data_test.split('/')[-1].replace('.gml','.txt')
        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        result_file = '%s/%s' %(save_dir_local, test_name)
        # g = nx.read_edgelist(data_test)
        g = nx.read_gml(data_test, destringizer=int)
        # g = self.Real2networkx(g_temp)
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            print ('number of nodes:%d'%(nx.number_of_nodes(g)))
            print ('number of edges:%d'%(nx.number_of_edges(g)))
            if stepRatio > 0:
                step = int(stepRatio*nx.number_of_nodes(g)) #step size
            else:
                step = 1
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            solution = self.GetSolution(0,step)
            t2 = time.time()
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                f_out.write('%d\n' % solution[i])
        self.ClearTestGraphs()
        return solution, solution_time

    def EvaRealForModelSelect(self, model_file, data_test, stepRatio=0.0025):  #测试真实数据
        g = nx.read_gml(data_test, destringizer=int)
        g_inner = self.GenNetwork(g)
        if stepRatio > 0:
            step = int(stepRatio*nx.number_of_nodes(g)) #step size
        else:
            step = 1
        self.InsertGraph(g, is_test=True)
        sol = self.GetSolution(0,step)
        self.ClearTestGraphs()
        nodes = list(range(nx.number_of_nodes(g)))
        sol_left = list(set(nodes)^set(sol))
        solution = sol + sol_left
        Robustness = self.utils.getRobustness(g_inner, solution)
        return Robustness

    def GetSolution(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        start = time.time()
        cdef int iter = 0
        cdef int new_action
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        return sol


    def EvaluateSol(self, data_test, sol_file, strategyID=0, reInsertStep=20):
        #evaluate the robust given the solution, strategyID:0,count;2:rank;3:multipy
        sys.stdout.flush()
        # g = nx.read_weighted_edgelist(data_test)
        g = nx.read_gml(data_test, destringizer=int)
        g_inner = self.GenNetwork(g)
        print ('number of nodes:%d'%nx.number_of_nodes(g))
        print ('number of edges:%d'%nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print ('number of sol nodes:%d'%len(sol))
        sol_left = list(set(nodes)^set(sol))
        if strategyID > 0:
            start = time.time()
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, reInsertStep)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print ('number of solution nodes:%d'%len(solution))
        Robustness = self.utils.getRobustness(g_inner, solution)
        MaxCCList = self.utils.MaxWccSzList
        return Robustness, MaxCCList, solution


    def Test(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        cdef int i
        sol = []
        while (not self.test_env.isTerminal()):
            # cost += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness


    def GetSol(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        # start = time.time()
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        # end = time.time()
        # print ('solution obtained time is:%.8f'%(end-start))
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        # print ('Robustness is:%.8f'%(Robustness))
        return Robustness, sol

    def GetSolReinsert(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        # start = time.time()
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        nodes = list(range(g_list[0].num_nodes))
        sol_left = list(set(nodes)^set(sol))
        sol_reinsert = self.utils.reInsert(g_list[0], sol, sol_left, 1, 1)
        solution = sol_reinsert + sol_left
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness, sol

    def SaveModel(self,model_path):
        torch.save(self.FINDER_net.state_dict(), model_path)
        print('model has been saved success!')

    def LoadModel(self,model_path):
        try:
            self.FINDER_net.load_state_dict(torch.load(model_path))
        except:
            self.FINDER_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        print('restore model from file successfully')


    def loadGraphList(self, graphFileName):
        with open(graphFileName, 'rb') as f:
            graphList = cp.load(f)
        print("load graph file success!")
        return graphList


    def GenNetwork(self, g):    #networkx2four
        nodes = g.nodes()
        edges = g.edges()
        weights = []
        for i in range(len(nodes)):
            weights.append(g.nodes[i]['weight'])
            #weights.append(g.nodes[i]['weight'])
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
            W = np.array(weights)
        else:
            A = np.array([0])
            B = np.array([0])
            W = np.array([0])
        return graph.py_Graph(len(nodes), len(edges), A, B, W)


    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best

    def Isterminal(self, graph):
        if len(nx.edges(graph)) == 0:
            return True
        else:
            return False

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', 'HCA'
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(node)
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        return Robustness

    def Real2networkx(self, G):
        cc = sorted(nx.connected_components(G), key=len, reverse=True)
        lcc = cc[0]
        # remove all nodes not in largest component
        numrealnodes = 0
        node_map = {}
        G_copy = G.copy()
        nodes = G_copy.nodes()
        weights = {}
        for node in nodes:
            if node not in lcc:
                G.remove_node(node)
                continue
            node_map[node] = numrealnodes
            weights[numrealnodes] = G_copy.node[node]['weight']
            numrealnodes += 1
        # re-create the largest component with nodes indexed from 0 sequentially
        g = nx.Graph()
        for edge in G.edges():
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            g.add_edge(src_idx, dst_idx)
        nx.set_node_attributes(g, 'weight', weights)
        return g

    def Evaluate4Vis(self, g, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        print ('testing')
        sys.stdout.flush()
        self.InsertGraph(g, is_test=True)
        sol = self.GetSol4Vis(0)
        print (sol)
        fout_sol = open('./Visualize/PKL/Sol.pkl','ab')
        cp.dump(sol, fout_sol)
        fout_sol.close()


    def GetSol4Vis(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        cdef int iter = 0
        while (not self.test_env.isTerminal()):
            fout_q = open('./Visualize/PKL/Q_%d.pkl'%iter,'ab')
            fout_embed = open('./Visualize/PKL/Embed_%d.pkl'%iter,'ab')
            list_pred, list_embed = self.Predict4Vis(g_list, [self.test_env.action_list])
            cp.dump(list_pred[0], fout_q)
            cp.dump(list_embed[0], fout_embed)
            fout_q.close()
            fout_embed.close()
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
            iter += 1
        return sol


    def Predict4Vis(self,g_list,covered):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            q_all, node_embed = self.session.run([self.q_on_all, self.node_embedding], feed_dict={
                self.rep_global: self.inputs['rep_global'],
                self.n2nsum_param: self.inputs['n2nsum_param'],
                self.subgsum_param: self.inputs['subgsum_param'],
                self.node_input: self.inputs['node_input'],
                self.aux_input: np.array(self.inputs['aux_input'])})

            raw_output = q_all
            raw_embed = node_embed
            # print("#####@!@@@")
            # print(np.shape(node_embed[0]))
            pos = 0
            pred = []
            embed = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                cur_embed = []
                for m in range(len(idx_map)):
                    cur_embed.append([])
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                        cur_embed[k] = []
                    else:
                        cur_pred[k] = raw_output[pos]
                        cur_embed[k] = raw_embed[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                    cur_embed[k] = []
                pred.append(cur_pred)
                embed.append(cur_embed)
            assert (pos == len(raw_output))
        return pred,embed
