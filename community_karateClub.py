# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:54:34 2022

@author: HP
"""
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so
import networkx as nx

import Random_walker_network as Rwn
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.preprocessing import normalize

#Let's import the network
N = nx.karate_club_graph()



#Let's calculate the adjacency matrix
A = nx.convert_matrix.to_numpy_matrix(N)

A = np.array(A, dtype=np.float64)
A_normalized = normalize(A, norm='l1', axis=1) 

#Let's evaluate the degree matrix D
D = np.diag(np.sum(A, axis=0))

#Let's evaluate the transition matrix T
T = np.dot(np.linalg.inv(D), A)


#Let's define the attributes of the Markov_Graph class
trans_m=A_normalized
n_vertex=trans_m.shape[0]
absorption = np.ones(n_vertex)*0.3
not_markov = False
starting_p=np.append(1, np.zeros(n_vertex-1))#the first node is set as the starting point of the walker

#Let's import the Markov_Graph class
G = Rwn.Markov_Graph(trans_m, absorption, not_markov, starting_p)
q=trans_m[4]

def simu(initial_node, current_node):
    """
    First of all, the function checks if the walker is absorbed using the G.absorbed() function.
    If it is true, 
    Then, starting from the initial_node, it performs the diffusion 
    of the walker: it jumps from the current_node to next one according to the G.sim_step() function.
    

    Parameters
    ----------
    initial_node : int
        The initial node of the simulation.
    current_node : int
        The node where the walker is.

    Returns
    -------
    G.visited_nodes : array
    
    For each step of the simulation this function updates the number of times the walker 
    has been in each nodes.
    

    """
    current_node=G.initialize(initial_node)
    for i in range(100):
        if G.absorbed(current_node):
            current_node = G.initialize(initial_node)
        else:
            current_node =G.sim_step(current_node)
    return G.visited_nodes

vec=[]

#Let's apply the function sim() starting from each node of the network
for i in range(G.n_nodes):
    initial_node=i
    current_node = G.initialize(initial_node)
    
    for k in range(100):   
        status=simu(initial_node, current_node)
        sum_status =+ status 
        print(i, k)
    
    mean_status = sum_status/100
    vec.append(mean_status)


      

#vec = np.append(vec, mean_status)
#Let's create a matrix which contains in each row (according to the different initial node) the G.visited_nodes array
#vec2matrix=vec.reshape(G.n_nodes, G.n_nodes)
#print(vec2matrix)

df = pd.DataFrame(vec, columns=["N{}".format(i) for i in range(G.n_nodes)])


#sns.clustermap(vec)  
#Let's compute the distance matrix 
Y = pdist(df.values, 'euclidean')

#Let's create the linkage matrix and the dendogram
Z = linkage(Y, 'ward', optimal_ordering=True)

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, show_leaf_counts=True)
plt.axhline(y = 25, color = 'r', linestyle = '-')
plt.show()   

#Let's form flat clusters from the hierarchical clustering defined by the given linkage matrix
X=fcluster(Z, t=25, criterion='distance', depth=2, R=None, monocrit=None)
print(X)

#The number of clusters
k=max(X) 
print(k)



#Let's print the nodes which belong to the same cluster
for i in range(1, k+1):
    cl_i=np.where(X==i)
    print(cl_i[0])
    print(len(cl_i[0]))
    
color_map = []

color=('red', 'green', 'yellow', 'blue', 'purple', 'pink', 'white')


for s in range(1,k+1):
     for i in range(G.n_nodes):
         if X[i]==s:
             color_map.append(color[s])
         continue
     
initial_pos=list(N.nodes)
layout_pos = nx.circular_layout(N)        
nx.draw(N, pos=layout_pos, node_color=color_map, with_labels=True)
plt.show() 
   
