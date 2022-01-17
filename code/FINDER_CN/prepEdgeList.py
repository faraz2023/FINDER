import networkx as nx
import numpy as np
import random
import os

data_path = os.path.join("..", "..", "data")
#data_name = ['ba_space_100']
data_name = ['day20']
save_dir = os.path.join("..", "..", "data")

i = 0
data = os.path.join(data_path , data_name[i] + '.edgelist')
g = nx.read_edgelist(data)

nodes = g.nodes()
nodes_l = list(nodes)
nodes_l_map = map(int, nodes_l)
nodes_l_int = list(nodes_l_map)
nodes_l_int.sort()

nodes_l_map = map(str, nodes_l_int)
nodes_l = list(nodes_l_map)

new_node_labels = {}
for i in range(len(nodes_l)):
    new_node_labels[nodes_l[i]] = str(i)

new_g = nx.relabel_nodes(g, new_node_labels)

nx.write_edgelist(new_g, os.path.join(save_dir, "day20_modified.edgelist"))
'''# adding node ids that do not exist (people no longer in the communities)
for j in range(nx.number_of_nodes(g)):
    if not str(j) in g.nodes():
        g.add_node(str(j))
'''
#nx.write_edgelist(g, os.path.join(save_dir, "day20_modified.edgelist"))