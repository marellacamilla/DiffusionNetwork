# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:26:26 2023

@author: HP
"""
import numpy as np
import networkx as nx 
import networkcluster as nc
import sys
from sys import argv
import configparser
import pandas as pd




config = configparser.ConfigParser()
config.read(sys.argv[1])

it = config.get('settings', 'it')
steps = config.get('settings', 'steps')
process_type = config.get('settings', 'process_type')
absorption_prob = config.get('settings', 'absorption_prob')
t = config.get('settings', 't')


it = int(it)
steps = int(steps)
absorption_prob=float(absorption_prob)
t=int(t)

#Build the network from links' file
df=pd.read_csv(sys.argv[2])
G = nx.from_pandas_edgelist(df, source='ID1', target='ID2',
                            edge_attr='LINK-TYPE')


#Transition Matrix
trans_matrix=nc.trans_matrix(G)

#Diffusion on the network: outcomes of the random_walk function are saved.
with open('random_walk_data.npy', 'wb') as f:  
   
    rw = nc.random_walk(it, steps, process_type, absorption_prob, trans_matrix)
    np.save(f, rw)

    
          