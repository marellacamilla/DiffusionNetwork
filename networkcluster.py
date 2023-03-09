# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:28:06 2022

@author: HP
"""

import numpy as np
import random
import networkx as nx
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def trans_matrix(N):
    """
    This function calculates the transition matrix associated to the network N.

    Parameters
    ----------
    N : networkx.classes.graph
        

    Returns
    -------
    trans_m : sparse._csr.csr_matrix
        

    """
    
    A=nx.adjacency_matrix(N)
    trans_m = normalize(A, norm='l1', axis=1)
    trans_m = A
    return trans_m
 

def initialize(initial_node, visited_nodes):
    """
    This function registers the initial position of the walker, incrementing by 
    one the number of times the walker has been in that node in the 
    visited_nodes array.

    Parameters
    ----------
    initial_node : int
        It is the starting point, where the walker is at the 
        beginning.
        
    visited_nodes : array
        Each entry defines the number of times the walker has been in each node.

    Returns
    -------
    initial_node : int
        

    """  
    
    visited_nodes[initial_node] += 1
    return initial_node

def sim_step(current_node, process_type, trans_m, visited_nodes):
    """
    This function, given the current position of the walker, defines where the 
    walker will jump, so the next_node of the walk, and increases by 
    one the number of times the walker has been in that node in the 
    visited_nodes array.
    

    Parameters
    ----------
    current_node : int
        The node where the walker is.
    process_type : string
        This tells if the process is Markovian or not.
    visited_nodes : array
        The number of times the walker has been in each node.
    trans_m : sparse._csr.csr_matrix
       Transition matrix
       
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
        next_node=random.choices(np.arange(0,trans_m.shape[0]), weights=p, k=1)
        visited_nodes[next_node] += 1
        return next_node
    
def absorbed(absorption_prob, current_node, trans_m):
    """
    The function evaluates the probability of absorption at the current_node.
    It returns True if the walker is absorbed, False if it is not.

    Parameters
    ----------
   absorption_prob: float
      The probability that the walker is absorbed in that node.
    current_node : int
        The node where the walker is.
    trans_m : sparse._csr.csr_matrix
       Transition matrix

    Returns
    -------
    Bool
        True: the walker is absorbed.
        False: the walker is not absorbed

    """
    
    
    absorption =random.choices([1,0], weights= [absorption_prob, 1-absorption_prob], k=1 )
    if absorption[0] == 1:
        return 1
    return 0

def simulation(initial_node, steps, process_type, absorption_prob, trans_m):
    """
    This function checks if the walker is absorbed in the node where it is.
    If it is absorbed, it will return in the initial position, otherwise the function 
    sim_step will evaluate the next node of the diffusion.
    If the process is not Markovian then the trasition probability at the current node is set to 0.
    The walker cannot return where it just was.

    Parameters
    ----------
    initial_node : int
        The initial node of the simulation.
    steps : int
        The number of steps that the walker does.
    process_type: string
        This tells if the process is Markovian or not.
    absorption_prob: float
       The probability that the walker is absorbed in that node.
    trans_m : sparse._csr.csr_matrix
       Transition matrix


    Returns
    -------
    visited_nodes : array
        The number of times the walker has been in each node during the simulation.

    """
    visited_nodes=np.array([0 for i in range(trans_m.shape[0])])
    current_node=initialize(initial_node, visited_nodes) #to mark that the node has been in the initial node
    for i in range(steps):
        if absorbed(absorption_prob, current_node, trans_m)==1:
            current_node = initialize(initial_node, visited_nodes) #If it is absorbed, It will re-start from the initial position
        else:
            current_node =sim_step(current_node, process_type, trans_m, visited_nodes)
    return visited_nodes


def random_walk(it, steps, process_type, absorption_prob, trans_m):
    """
    The function performs the simulation considering each node of the network
    as the initial_node.
    The simulation is repeated it times and the averages of the outcomes are calculated. 

    Parameters
    ----------
    it : int
        The number of times the simulation is repeated for each staring point 
        for calculating the average values of the outcomes.
    steps : int
        The number of steps that the walker does.
    process_type: string
        This tells if the process is Markovian or not.
    absorption_prob: float
        The probability that the walker is absorbed in that node.
    trans_m : sparse._csr.csr_matrix
        Transition matrix

    Returns
    -------
    vec : ndarray
        A matrix, in which each row is the outcome of the simulation function 
        considering as the intiial node the one corresponding to the number of that row, 
        averaged over the number of simulations.

    """

    
    vec=np.zeros([trans_m.shape[0], trans_m.shape[0]], float)
    for i in range(trans_m.shape[0]):
        initial_node=i
        for k in range(it):   
            entrances=simulation(initial_node, steps, process_type, absorption_prob, trans_m)
            
            sum_entrances =+ entrances 
            
            
        mean_entrances = sum_entrances/it
        
        vec[i] = mean_entrances
    
    return vec
    
def linkage_matrix(data):
    """
    The function computes the distance matrix (Euclidian distance) and 
    the Linkage matrix using the method 'Ward'. 

    Parameters
    ----------
    data : ndarray
        Outcome of the function random_walk(it, steps, process_type, absorption_prob, trans_m)


    Returns
    -------
    Z : ndarray
        The linkage matrix of the hierarchical clustering.

    """
    
    Y = pdist(data, 'euclidean')
    Z = linkage(Y, 'ward', optimal_ordering=True)    
    return Z

def plot_dendogram(t, link_m):
    """
    This function generates the dendogram which represents the hierarchical clustering

    Parameters
    ----------
    t : int
        Threshold of the hierarchical clustering.
    link_m : linkage_matrix(it, steps, process_type, absorption_prob, trans_m)

    Returns
    -------
    The dendrogram.

    """
    
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(link_m, show_leaf_counts=True)
    plt.axhline(y = t, color = 'r', linestyle = '-')
    plt.show() 
    
def clustering(t, link_m):
    """
    This function calculates the flat clusters from the hierarchical clustering
    defined by the given linkage matrix

    Parameters
    ----------
    t : int
        Threshold of the hierarchical clustering.
    link_m : ndarray
       Linkage_matrix(it, steps, process_type, absorption_prob, trans_m)

    Returns
    -------
    X : array
        Each element of this array represents the node associated to the index's number.
        The array contains the number of the cluster the corresponding 
        node belongs to.

    """
    
    X=fcluster(link_m, t=t, criterion='distance', depth=2, R=None, monocrit=None)
    return X

def plot_clustering(N, t, X, trans_m):
    """
    This function generates the plot of the network, highlighting the nodes 
    with different colors according to the clusters they belong to.

    Parameters
    ----------
    t : int
        Threshold of the hierarchical clustering.
    X : array
        clustering(t, it, steps, process_type, absorption_prob, trans_m)

    Returns
    -------
    Plot.

    """
    color=('red', 'blue', 'yellow', 'green', 'purple', 'pink', 'white')
    
    k=max(X)
    color_map=[]
    for s in range(1,k+1):
         for i in range(trans_m.shape[0]):
             if X[i]==s:
                 color_map.append(color[s])
             continue
         

    layout_pos = nx.spring_layout(N)        
    nx.draw(N, pos=layout_pos, node_color=color_map, with_labels=True)
    plt.show()
