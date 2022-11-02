import random 

import Random_walker_network as Rwn

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so



trans_m = np.array([[0.2,0.4,0.4],[0.5,0.,0.5],[0.1,0.9,0.]]) #unscaled
trans_m = trans_m/trans_m.sum(axis=1)[:,None] #normalization

n_vertex=trans_m.shape[0]
absorption = np.array([0.0, 0.0, 0.0])
not_markov = True
starting_p=[1, 0, 0]



G = Rwn.Markov_Graph(trans_m, absorption, not_markov, starting_p)




def simu(initial_node, current_node):
    current_node=G.initialize(initial_node)
    for i in range(100000):
        if G.absorbed(current_node):
            current_node = G.initialize(initial_node)
        else:
            current_node =G.sim_step(current_node)
            return G.visited_nodes
    
vec=[] 

for i in range(G.n_nodes):
     
    initial_node=i
    current_node = G.initialize(initial_node)
    #definisco quante volte ogni nodo è stato visitato       
    status=simu(initial_node, current_node)
    vec = np.append(vec, status)
    print(status)
    norm_status=status/G.steps



print(vec)

vec2matrix=vec.reshape(G.n_nodes, G.n_nodes)

print(vec2matrix)

sns.clustermap(vec2matrix)