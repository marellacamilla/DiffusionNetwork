# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:28:06 2022

@author: HP
"""

import numpy as np
import random
import pandas as pd
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib as mpl
import matplotlib.pyplot as plt

N=nx.karate_club_graph()
absorption_prob=0.4
process_type = "Markov"
    

def trans_matrix(N):
    """
    This function calculates the transition matrix associated to the network

    Parameters
    ----------
    N : networkx.classes.graph
        

    Returns
    -------
    trans_m : sparse._csr.csr_matrix
        

    """
    
    A=nx.adjacency_matrix(N)
    trans_m = normalize(A, norm='l1', axis=1)
    return trans_m

trans_m = trans_matrix(N)  
n_nodes = trans_m.shape[0] 
absorption = np.ones(n_nodes)*absorption_prob 
visited_nodes = np.array([0 for i in range(n_nodes)])  

def initialize(initial_node):
    """
    This function registers the initial position of the walker, incrementing by 
    one the number of times the walker has been in that node in the 
    visited_nodes array.

    Parameters
    ----------
    initial_node : int
        It is the starting point of the simulation, where the walker is at the 
        beginning.

    Returns
    -------
    initial_node : int
        

    """
    visited_nodes[initial_node] += 1
    return initial_node

def sim_step(current_node):
    """
    This function, given the current position of the walker, defines where the 
    walker will jump, so the next_node of the walk. 

    Parameters
    ----------
    current_node : int
        The node where the walker is.

    Returns
    -------
    next_node : int
        The node where the walker will jump.

    """
    
    
    
    p=np.array(trans_m[current_node, :].todense()).ravel()
    
    if process_type == "Markov":
        next_node =random.choices(np.arange(0, trans_m.shape[0]), weights=p, k=1)
        visited_nodes[next_node] += 1
        return next_node
    else:
        p[current_node]=0
        next_node=np.random.choice(np.arange(0,trans_m.shape[0]), p=p)
        visited_nodes[next_node] += 1
        return next_node
    
def absorbed(current_node):
    """
    The function evaluates the probability of absorption at the current_node.
    It returns True if the walker is absorbed, False if it is not.

    Parameters
    ----------
    current_node : int
        The node where the walker is.

    Returns
    -------
    Bool
        True: the walker is absorbed.
        False: the walker is not absorbed

    """
    
    p=absorption[current_node]
    prob_absorb = np.random.choice([1,0], 1, [p, 1-p])
    if prob_absorb == 1:
        return 1
    return 0

def simulation(initial_node):
    """
    This function checks if the walker is absorbed in the node where it is.
    If it is absorbed, it will return in the initial position, otherwise the function 
    sim_step will evaluate the the next node of the diffusion.

    Parameters
    ----------
    initial_node : int
        The initial node of the simulation.

    Returns
    -------
    visited_nodes : array
        The number of times the walker has been in each node during the simulation.

    """
    current_node=initialize(initial_node) #to mark that the node has been in the initial node
    for i in range(100):
        if absorbed(current_node):
            current_node = initialize(initial_node) #If it is absorbed, It will re-start from the initial position
        else:
            current_node =sim_step(current_node)
    return visited_nodes


def random_walk():
    """
    The function performs the simulation considering each node of the network
    as the initial_node. 

    Returns
    -------
    vec : array
        A matrix, in which each row is the outcomes of the simulation function 
        considering as the intiial node the one corresponding to the numer of that row.

    """
    vec=np.zeros([n_nodes, n_nodes], float)
    for i in range(n_nodes):
        initial_node=i
        for k in range(100):   
            entrances=simulation(initial_node)
            
            sum_entrances =+ entrances 
            
            
        mean_entrances = sum_entrances/100
        
        vec[i] = mean_entrances
    
    return vec
    
def linkage_matrix():
    """
    

    Returns
    -------
    Z : array
        The linkage matrix of the hierarchical clustering.

    """
    data=random_walk()
    Y = pdist(data, 'euclidean')
    #Let's create the linkage matrix and the dendogram
    Z = linkage(Y, 'ward', optimal_ordering=True)    
    return Z

def plot_dendogram(t):
    """
    This function generates the dendogram which represents the hierarchical clustering

    Parameters
    ----------
    t : int
        Threshold of the hierarchical clustering.

    Returns
    -------
    The dendrogram.

    """
    link_m=linkage_matrix(t)
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(link_m, show_leaf_counts=True)
    plt.axhline(y = t, color = 'r', linestyle = '-')
    plt.show() 
    
def clustering(t):
    """
    This function calculates the flat clusters from the hierarchical clustering
    defined by the given linkage matrix

    Parameters
    ----------
    t : int
        Threshold of the hierarchical clustering.

    Returns
    -------
    X : array
        Each element of this array represents the node associated to the index's number.
        The array contains the number of the cluster the corresponding 
        node belongs to.

    """
    link_m=linkage_matrix(t)
    X=fcluster(link_m, t=t, criterion='distance', depth=2, R=None, monocrit=None)
    return X

def plot_clustering(t):
    """
    This function generates the plot of the network, highlighting the nodes 
    with different colors according to the clusters they belong to.

    Parameters
    ----------
    t : int
        Threshold of the hierarchical clustering.

    Returns
    -------
    Plot.

    """
    color=('red', 'green', 'yellow', 'blue', 'purple', 'pink', 'white')
    X=clustering(t)
    k=max(X)
    color_map=[]
    for s in range(1,k+1):
         for i in range(n_nodes):
             if X[i]==s:
                 color_map.append(color[s])
             continue
         
    initial_pos=list(N.nodes)
    layout_pos = nx.circular_layout(N)        
    nx.draw(N, pos=layout_pos, node_color=color_map, with_labels=True)
    plt.show()