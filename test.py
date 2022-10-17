# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:37:28 2022

@author: Camilla Marella
"""
import Random_walker_network as rwn
import numpy as np
import simulation
%%file simualtion.py

def test_not_Markov_2_states():
    not_markov = True
    trans_m=([0,1],[1,0])
    absorption = np.array([0,0])
    starting_p = [1,0]
    simulation.status = status
    assert status == [1,0]
    
