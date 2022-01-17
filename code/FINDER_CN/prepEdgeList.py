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
print(nodes)
'''# adding node ids that do not exist (people no longer in the communities)
for j in range(nx.number_of_nodes(g)):
    if not str(j) in g.nodes():
        g.add_node(str(j))
'''
#nx.write_edgelist(g, os.path.join(save_dir, "day20_modified.edgelist"))