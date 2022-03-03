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
import torch
# from gurobipy import *

# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 1000000
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
cdef int max_bp_iter = 3
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage



class FINDER:

    def __init__(self):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type =  'barabasi_albert'#'barabasi_albert', 'small-world'
        self.training_type = 'degree'
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
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


    def Train(self):
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

        save_dir = './models/TORCH-Model_{}'.format(self.g_type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '{}/ModelVC_{}_{}.csv'.format(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            #start = time.clock()
            start = time.perf_counter()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

            if iter % 10 == 0:
                self.PlayGame(10, eps)
            #if iter % 300 == 0: #put this back for proper snapshot
            if iter % 300 == 300:
                if(iter == 0):
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
                #N_end = time.clock()
                N_end = time.perf_counter()
                print ('300 iterations total time: %.8fs'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            self.Fit()
        f_out.close()

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
            result_betweeness += self.HXA(g_betweenness,'HBA')
            self.InsertGraph(g, is_test=True)
        print ('Validation of HDA: %.16f'%(result_degree / n_valid))
        print ('Validation of HBA: %.16f'%(result_betweeness / n_valid))

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
        ### random weight
        if self.training_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0,1)
        # ### degree weight
        elif self.training_type == 'degree':
            degree = nx.degree(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]/maxDegree
        elif self.training_type == 'degree_noise':
            degree = nx.degree(g)
            #maxDegree = max(dict(degree).values())
            mu = np.mean(list(dict(degree).values()))
            std = np.std(list(dict(degree).values()))
            weights = {}
            for node in g.nodes():
                episilon = np.random.normal(mu, std, 1)[0]
                weights[node] = 0.5*degree[node] + episilon
                if weights[node] < 0.0:
                    weights[node] = -weights[node]
            maxDegree = max(weights.values())
            for node in g.nodes():
                weights[node] = weights[node] / maxDegree

        nx.set_node_attributes(g, weights,'weight')
        return g

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

    def GenNetwork(self, g):    #networkx2four
        nodes = g.nodes()
        edges = g.edges()
        weights = []
        for i in range(len(nodes)):
            #weights.append(g.node[i]['weight'])
            weights.append(g.nodes[i]['weight'])
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

    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step):
        cdef int num_env = len(self.env_list)
        cdef int n = 0
        cdef int i
        # To bipass q-network for now
        eps = 1.1
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

    def PlayGame(self,int n_traj, double eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)

    def PredictWithCurrentQNet(self,g_list,covered):
        result = self.Predict(g_list,covered,False)
        return result
    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result

    def Predict(self,g_list,covered,isSnapSnot):
        """
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
                result = self.session.run([self.q_on_allT], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            else:
                result = self.session.run([self.q_on_all], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            raw_output = result[0]
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
        """
        return None

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

        #print("=======")
        #print(sample.g_list)
        #print(sample.list_s_primes)
        #print("========")
        ## THIS CODE IS here to create a toy list_pred (must be removed in working verion!)
        list_pred = []
        for i in range(BATCH_SIZE):
            list_pred.append([1])


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

    def fit(self,g_list,covered,actions,list_target):
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
            print("----From fit: (dense dimensions of input tensors")
            print("Action Select:" , self.inputs['action_select'].dense_shape)
            print("rep_global:" , self.inputs['rep_global'].dense_shape)
            print("n2nsum_param:" , self.inputs['n2nsum_param'].dense_shape)
            print("laplacian_param:" , self.inputs['laplacian_param'].dense_shape)
            print("subgsum_param:" , self.inputs['subgsum_param'].dense_shape)
            print("(list) node_input:" , len(self.inputs['node_input']))
            print("(list) aux_input:" , len(self.inputs['aux_input']))
            print("(list) target:" , len(self.inputs['target']))
            result = self.session.run([self.loss,self.trainStep],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param : self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.aux_input: np.array(self.inputs['aux_input']),
                                        self.target : self.inputs['target']})
            loss += result[0]*bsize
        return loss / len(g_list)

    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat

    def TakeSnapShot(self):
        pass
        #self.session.run(self.UpdateTargetQNetwork)

    def SaveModel(self,model_path):
        #self.saver.save(self.session, model_path)
        #print('model has been saved success!')
        print("Complete this function to save the model")
        
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


