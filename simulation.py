import random 

import Random_walker_network as Rwn

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance
from scipy.cluster import hierarchy


trans_m = np.array([[0.2,0.4,0.4],[0.5,0.,0.5],[0.1,0.9,0.]]) #unscaled
trans_m = trans_m/trans_m.sum(axis=1)[:,None] #normalization

n_vertex=trans_m.shape[0]
absorption = np.array([0.0, 0.0, 0.0])
not_markov = True
starting_p=([1, 0, 0])

G = Rwn.Markov_Graph(trans_m, absorption, not_markov, starting_p)


G = Rwn.Markov_Graph(trans_m, absorption, not_markov, starting_p)

def simu(initial_node):
    
    current_node=G.initialize(initial_node)
    for i in range(100000):
        if G.absorbed(current_node):
            current_node = G.initialize(initial_node)
            current_node = G.initialize(initial_node)
        else:
            current_node =G.sim_step(current_node)
            return G.visited_nodes
    
def outcomes(G):
    vec=[]
    #Let's apply the function sim() starting from each node of the network
    for i in range(G.n_nodes):
        initial_node=i
        for k in range(100):   
            status=simu(initial_node)
            sum_status =+ status 
        
        mean_status = sum_status/100
        vec.append(mean_status)


    df = pd.DataFrame(vec, columns=["N{}".format(i) for i in range(G.n_nodes)])
    return (df)

def n_clusters(df):
    Y = pdist(df.values, 'euclidean')

    #Let's create the linkage matrix and the dendogram
    Z = linkage(Y, 'ward', optimal_ordering=True)
 

    #Let's form flat clusters from the hierarchical clustering defined by the given linkage matrix
    X=fcluster(Z, t=25, criterion='distance', depth=2, R=None, monocrit=None)
    

    #The number of clusters
    k=max(X) 
    return(k)

    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, show_leaf_counts=True)
    plt.axhline(y = 25, color = 'r', linestyle = '-')
    plt.show()
    
    return(plt.show())
   
    