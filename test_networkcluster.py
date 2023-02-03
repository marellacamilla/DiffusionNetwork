# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:37:28 2022

@author: Camilla Marella
"""
import numpy as np
import scipy
from hypothesis import given
import hypothesis.strategies as st
import pytest
import networkx as nx 
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import networkcluster as nc
from numpy.linalg import eig
import random
from hypothesis import given
import hypothesis.strategies as st






def test_all_rows_sum_up_to_one():
    """
    This function tests that the trans_matrix produces a matrix
    with all the rows normalized to .

    Returns
    -------
    None.

    """
    N=nx.Graph()
    
    N.add_edge("a", "b", weight=0.6)
    N.add_edge("a", "c", weight=0.2)
    N.add_edge("c", "b", weight=0.1)
    N.add_edge("c", "a", weight=0.7)
    N.add_edge("b", "a", weight=0.9)
    N.add_edge("b", "c", weight=0.3)
    M=nc.trans_matrix(N)
    for i in range(M.shape[0]):
        assert  np.sum(M[i, :])==1
        
def test_initialize():
    """
    This function checks that if the initial node is the number 2
    the function "initialize" increases by one the entry of the visited_nodes vector 
    at the position 2.

    Returns
    -------
    None.

    """
    initial_node=2
    visited_nodes=np.array([0 for i in range(5)])
    nc.initialize(initial_node, visited_nodes)
    assert visited_nodes[initial_node] == 1
    
def test_sim_step():
    """
    This function checks that the sim_step function increments by one
    the number of entries in the visited_nodes vector given 
    a simple network of two nodes. 

    Returns
    -------
    None.

    """
    N=nx.Graph()
    
    N.add_edge("a", "b", weight=0.5)
    N.add_edge("a", "a", weight=0.5)
    N.add_edge("b", "b", weight=0.5)
    N.add_edge("b", "a", weight=0.5)
    trans_m = nc.trans_matrix(N)
    visited_nodes=np.array([0 for i in range(2)])
    
    nc.sim_step(0, "Markov", trans_m, visited_nodes)
    entries=np.sum(visited_nodes)
    assert entries == 1
 
    
 
def test_absorption():
    """
    This function checks that if the absorption probability is 1, the "absorbed" function
    gives as output 1 (so the walker is absorbed).

    Returns
    -------
    None.

    """
    trans_m = nc.trans_matrix(nx.karate_club_graph())
    a = nc.absorbed(1, 2, trans_m)
    print(a)
    assert a == 1
    
        
     
def test_simulation_absorption_prob_one():
    """
    This function checks that if the absorption_prob is 1 the walker will stay
    in the initial_node. So in the visited_nodes vector, the entry correspondent to
    the initial_node position si equal to the number of steps (+1 for the initialize).

    Returns
    -------
    None.

    """
    N=nx.Graph()
    
    N.add_edge("a", "b", weight=1)
    N.add_edge("a", "c", weight=0)
    N.add_edge("c", "b", weight=0.1)
    N.add_edge("c", "a", weight=0.7)
    N.add_edge("b", "a", weight=0.9)
    N.add_edge("b", "c", weight=0.3)
    trans_m = nc.trans_matrix(N)
    
    
    process_type= "NotMarkov"
    initial_node=0
    absorption_prob=1
    steps=15
    entries = nc.simulation(initial_node, steps, process_type, absorption_prob, trans_m)
    
    #+1 because the simulation function initialize the initial_node (increasing by one the entry in visited_node)
    assert entries[0]==steps+1
    
    
   
def test_stationary_distribution(steps):
    """
    Considering a simple network made of three nodes, the function comuptes the stationary 
    distribution like an iterative product between the transition matrix and the initial distribution [1, 0, 0].
    Then it computes the stationaty distribution through the simulation function.
    It checks that the difference of each entry of this two array is <0.1.

    Parameters
    ----------
    steps : int
        Number of jumps of the walker and number of products between the transition matrix
        and the distribution vector.

    Returns
    -------
    None.

    """
    #Let's build a simple network, composed of 3 nodes
    N=nx.Graph()
    
    N.add_edge("a", "b", weight=0.6)
    N.add_edge("a", "c", weight=0.2)
    N.add_edge("c", "b", weight=0.1)
    N.add_edge("c", "a", weight=0.7)
    N.add_edge("b", "a", weight=0.9)
    N.add_edge("b", "c", weight=0.3)
    #Its transition matrix is
    trans_m = nc.trans_matrix(N)
    
    absorption_prob=0
    process_type="Markov"
    initial_node=0
    
    random.seed(0)
    #stationary distribution: normalized number of times the walekr has been in each node.
    s_state=nc.simulation(initial_node, steps, process_type, absorption_prob, trans_m)/steps
    
    #stationary distribution: iterative product between the initial distribution and the trans_m.
    state=np.array([1, 0, 0])#Because initial_node=0
    P=trans_m.toarray()
    for i in range(steps):
        state=np.dot(state,P)
    print(state)
    print(s_state)
    assert s_state[0] - state[0] <0.1
    assert s_state[1] - state[1] <0.1
    assert s_state[2] - state[2] <0.1
   
def test_Not_Markov():
    """
    This function verifies that if the process in not Markovian, considering a simple
    chain of 2 nodes, if the walker does 3 steps (jumps) starting from one node 
    the visited_nodes array registers 2 times the walker in each entry.
    This because the walker can just jump in the other node.

    Returns
    -------
    None.

    """
    N=nx.Graph()
    
    N.add_edge("a", "b", weight=0.5)
    N.add_edge("a", "a", weight=0.5)
    N.add_edge("b", "b", weight=0.5)
    N.add_edge("b", "a", weight=0.5)
    
    #Its transition matrix is
    trans_m = nc.trans_matrix(N)
    visited_nodes=nc.simulation(0, 3, "NotMarkov",0, trans_m)#The probability of absorption is zero.
    assert visited_nodes[0] == 2
    assert visited_nodes[1] == 2