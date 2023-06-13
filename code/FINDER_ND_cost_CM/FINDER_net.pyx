import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse

#out = torch_sparse.spmm(index, value, 3, 3, matrix)

class FINDER_net(nn.Module):
    def __init__(self, embedding_size=64, w_initialization_std=1, reg_hidden=32, max_bp_iter=3, embeddingMethod=1, aux_dim=4, device=None):
        super(FINDER_net, self).__init__()
        
        self.rand_generator = torch.normal
        self.embedding_size = embedding_size
        self.w_initialization_std = w_initialization_std
        self.reg_hidden = reg_hidden
        self.max_bp_iter = max_bp_iter
        self.embeddingMethod = embeddingMethod
        self.aux_dim = aux_dim
        self.device = device
        
        self.act = nn.ReLU()
        
        # [2, embed_dim]
        self.w_n2l = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(2, self.embedding_size)))
        # [embed_dim, embed_dim]
        self.p_node_conv = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.embedding_size, self.embedding_size)))
        
        if self.embeddingMethod == 1:    #'graphsage'
            # [embed_dim, embed_dim]
            self.p_node_conv2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.embedding_size, self.embedding_size)))
            # [2*embed_dim, embed_dim]
            self.p_node_conv3 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(2*self.embedding_size, self.embedding_size)))

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            # [embed_dim, reg_hidden]
            #h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.embedding_size, self.reg_hidden)))
            
            #[reg_hidden+aux_dim, 1]
            #h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            self.h2_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.reg_hidden + self.aux_dim, 1)))
            #[reg_hidden2 + aux_dim, 1]
            self.last_w = self.h2_weight
        else:
            #[2*embed_dim, reg_hidden]
            #h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(2*self.embedding_size, self.reg_hidden)))            
            #[2*embed_dim, reg_hidden]
            self.last_w = self.h1_weight
            
        ## [embed_dim, 1]
        #cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)
        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.embedding_size, 1)))


    def train_forward(self, node_input, subgsum_param, n2nsum_param, action_select, aux_input):
        
        y_nodes_size = subgsum_param['m']
        # [batch_size, 2]
        y_node_input = torch.ones((y_nodes_size,2)).type(torch.FloatTensor).to(self.device)

        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        # no sparse
        #input_message = tf.matmul(tf.cast(self.node_input,tf.float32), w_n2l)
        input_message = torch.matmul(node_input, self.w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        #input_potential_layer = tf.nn.relu(input_message)
        input_potential_layer = self.act(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        y_input_message = torch.matmul(y_node_input, self.w_n2l)
        #[batch_size, embed_dim]  # no sparse
        #y_input_potential_layer = tf.nn.relu(y_input_message)
        y_input_potential_layer = self.act(y_input_message)

        #input_potential_layer = input_message
        lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)

        # max_bp_iter=3  
        while lv < self.max_bp_iter:
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            #n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer)
            # see https://discuss.pytorch.org/t/sparse-tensors-in-pytorch/859/4
            #OLD n2npool = torch.matmul(n2nsum_param, cur_message_layer)
            n2npool = torch_sparse.spmm(n2nsum_param['index'], n2nsum_param['value'],\
                 n2nsum_param['m'], n2nsum_param['n'], cur_message_layer)
            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            #node_linear = tf.matmul(n2npool, p_node_conv)
            node_linear = torch.matmul(n2npool, self.p_node_conv)

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            #y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            # why cur_message_layer, instead of y_cur_message_layer?
            #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
            y_n2npool = torch_sparse.spmm(subgsum_param['index'], subgsum_param['value'],\
                 subgsum_param['m'], subgsum_param['n'], cur_message_layer)

            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            #y_node_linear = tf.matmul(y_n2npool, p_node_conv)
            y_node_linear = torch.matmul(y_n2npool, self.p_node_conv)

            if self.embeddingMethod == 0: # 'structure2vec'
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                #merged_linear = tf.add(node_linear,input_message)
                merged_linear = torch.add(node_linear,input_message)
                #[node_cnt, embed_dim]
                #cur_message_layer = tf.nn.relu(merged_linear)
                cur_message_layer = self.act(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                #y_merged_linear = tf.add(y_node_linear, y_input_message)
                y_merged_linear = torch.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                #y_cur_message_layer = tf.nn.relu(y_merged_linear)
                y_cur_message_layer = self.act(y_merged_linear)
                
            else:   # 'graphsage'
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                #cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
                cur_message_layer_linear = torch.matmul(cur_message_layer, self.p_node_conv2)
                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                #merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                #cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))
                cur_message_layer = self.act(torch.matmul(merged_linear, self.p_node_conv3))

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                #y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                y_cur_message_layer_linear = torch.matmul(y_cur_message_layer, self.p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                #y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                #y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))
                y_cur_message_layer = self.act(torch.matmul(y_merged_linear, self.p_node_conv3))

            #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)

        
        node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        y_potential = y_cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        #action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
        #OLD action_embed = torch.matmul(action_select, cur_message_layer)
        action_embed = torch_sparse.spmm(action_select['index'], action_select['value'],\
                 action_select['m'], action_select['n'], cur_message_layer)

        # # [batch_size, embed_dim, embed_dim]
        #temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1))
        temp = torch.matmul(torch.unsqueeze(action_embed, dim=2),torch.unsqueeze(y_potential, dim=1))
        # [batch_size, embed_dim]
        #Shape = tf.shape(action_embed)
        Shape = action_embed.size()
        # [batch_size, embed_dim], first transform
        #embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)
        embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product,[Shape[0],1]),\
                                                                   [Shape[0],Shape[1],1])),Shape)

        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            #hidden = tf.matmul(embed_s_a, h1_weight)
            hidden = torch.matmul(embed_s_a, self.h1_weight)
            #[batch_size, reg_hidden]
            #last_output = tf.nn.relu(hidden)
            last_output = self.act(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        #last_output = tf.concat([last_output, self.aux_input], 1)
        last_output = torch.concat([last_output, aux_input], 1)
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        #q_pred = tf.matmul(last_output, last_w)
        q_pred = torch.matmul(last_output, self.last_w)
        
        
        return q_pred, cur_message_layer
    
    def test_forward(self, node_input, subgsum_param, n2nsum_param, rep_global, aux_input):
        
        y_nodes_size = subgsum_param['m']
        # [batch_size, 2]
        y_node_input = torch.ones((y_nodes_size,2)).type(torch.FloatTensor).to(self.device)

        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        # no sparse
        #input_message = tf.matmul(tf.cast(self.node_input,tf.float32), w_n2l)
        input_message = torch.matmul(node_input, self.w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        #input_potential_layer = tf.nn.relu(input_message)
        input_potential_layer = self.act(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        y_input_message = torch.matmul(y_node_input, self.w_n2l)
        #[batch_size, embed_dim]  # no sparse
        #y_input_potential_layer = tf.nn.relu(y_input_message)
        y_input_potential_layer = self.act(y_input_message)

        #input_potential_layer = input_message
        lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)

        # max_bp_iter=3  
        while lv < self.max_bp_iter:
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            #n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer)
            # see https://discuss.pytorch.org/t/sparse-tensors-in-pytorch/859/4
            # OLD n2npool = torch.matmul(n2nsum_param, cur_message_layer)
            n2npool = torch_sparse.spmm(n2nsum_param['index'], n2nsum_param['value'],\
                 n2nsum_param['m'], n2nsum_param['n'], cur_message_layer)

            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            #node_linear = tf.matmul(n2npool, p_node_conv)
            node_linear = torch.matmul(n2npool, self.p_node_conv)

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            #y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            # why cur_message_layer, instead of y_cur_message_layer?
            #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
            y_n2npool = torch_sparse.spmm(subgsum_param['index'], subgsum_param['value'],\
                 subgsum_param['m'], subgsum_param['n'], cur_message_layer)

            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            #y_node_linear = tf.matmul(y_n2npool, p_node_conv)
            y_node_linear = torch.matmul(y_n2npool, self.p_node_conv)

            if self.embeddingMethod == 0: # 'structure2vec'
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                #merged_linear = tf.add(node_linear,input_message)
                merged_linear = torch.add(node_linear,input_message)
                #[node_cnt, embed_dim]
                #cur_message_layer = tf.nn.relu(merged_linear)
                cur_message_layer = self.act(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                #y_merged_linear = tf.add(y_node_linear, y_input_message)
                y_merged_linear = torch.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                #y_cur_message_layer = tf.nn.relu(y_merged_linear)
                y_cur_message_layer = self.act(y_merged_linear)
                
            else:   # 'graphsage'
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                #cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
                cur_message_layer_linear = torch.matmul(cur_message_layer, self.p_node_conv2)
                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                #merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                #cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))
                cur_message_layer = self.act(torch.matmul(merged_linear, self.p_node_conv3))

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                #y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                y_cur_message_layer_linear = torch.matmul(y_cur_message_layer, self.p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                #y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                #y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))
                y_cur_message_layer = self.act(torch.matmul(y_merged_linear, self.p_node_conv3))

            #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)

        
            y_potential = y_cur_message_layer        
            
        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        #OLD rep_y = torch.matmul(rep_global, y_potential)
        rep_y = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
                 rep_global['m'], rep_global['n'], y_potential)

        #[[node_cnt, embed_dim], [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim]
        # embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)
        # # [node_cnt, embed_dim, embed_dim]
        temp1 = torch.matmul(torch.unsqueeze(cur_message_layer, dim=2),torch.unsqueeze(rep_y, dim=1))
        # [node_cnt embed_dim]
        Shape1 = cur_message_layer.size()
        # [batch_size, embed_dim], first transform
        embed_s_a_all = torch.reshape(torch.matmul(temp1, torch.reshape(torch.tile(self.cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = torch.matmul(embed_s_a_all, self.h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = self.act(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]
            # last_output_hidden = tf.matmul(last_output1, h2_weight)
            # last_output = tf.nn.relu(last_output_hidden)

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
            rep_global['m'], rep_global['n'], aux_input)
        #rep_aux = torch.matmul(rep_global, aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = torch.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = torch.matmul(last_output, self.last_w)


        return q_on_all