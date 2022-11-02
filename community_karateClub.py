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

#Let's import the network
N = nx.karate_club_graph()

#Let's calculate the adjacency matrix
A = nx.convert_matrix.to_numpy_matrix(N)

A = np.array(A, dtype=np.float64)

#Let's evaluate the degree matrix D
D = np.diag(np.sum(A, axis=0))

#..the transition matrix T
T = np.dot(np.linalg.inv(D), A)


#Let's define the attributes of the Markov_Graph class
trans_m=T
n_vertex=trans_m.shape[0]
absorption = np.zeros(n_vertex)
not_markov = True
starting_p=np.append(1, np.zeros(n_vertex-1))


G = Rwn.Markov_Graph(trans_m, absorption, not_markov, starting_p)

def simu(initial_node, current_node):
    """
    This function starting from the initial_node performs the diffusion 
    of the walker: it jumps from the current_node to next one according to the G.sim_step() function.
    Before, the function checks if the walker is absorbed using the funcion G.absorbed().

    Parameters
    ----------
    initial_node : int
        The initial node of the simulation.
    current_node : int
        The node where the walker is.

    Returns
    -------
    For each step of the simulation this function updates the number of times the walker 
    has been in each nodes.
        

    """
    current_node=G.initialize(initial_node)
    for i in range(100000):
        if G.absorbed(current_node):
            current_node = G.initialize(initial_node)
        else:
            current_node =G.sim_step(current_node)
            return G.visited_nodes
    
vec=[]

#Let's apply the function sim() staring from each node of the network
for i in range(G.n_nodes):
   
    initial_node=i
    current_node = G.initialize(initial_node)
    #definisco quante volte ogni nodo è stato visitato       
    status=simu(initial_node, current_node)
    vec = np.append(vec, status)
    norm_status=status/G.steps
    
    
 

#Let's create a matrix which contains in each row the (according to the different initial node) the G.visited_vector
vec2matrix=vec.reshape(G.n_nodes, G.n_nodes)
print(vec2matrix)


sns.clustermap(vec2matrix)   

#Let's create the linkage matrix and the dendogram
Z = linkage(vec2matrix, 'single')#to produce the linkage matrix
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()   


X=fcluster(Z, t=1.1, criterion='inconsistent', depth=2, R=None, monocrit=None)
print(X)

#The number of clusters
k=max(X) 
print(k)