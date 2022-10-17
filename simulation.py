import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt
import Random_walker_network as Rwn


trans_m = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]]) #unscaled
trans_m = trans_m/trans_m.sum(axis=1)[:,None] #normalization

absorption = np.array([0.0, 0.0, 0.0])
not_markov = True

#The initial point of the simulation is the first node
starting_p = [1,0,0]

G=Rwn.Markov_Graph(trans_m, absorption, not_markov, starting_p)
current_node = G.initialize()

for i in range(20):
    if G.absorbed(current_node):
         print("The walker has been absorbed")
         current_node = G.initialize() #if the walker is absorbed the simulation is reinitialized
    continue
    current_node = G.sim_step(current_node)
    print(current_node)

status=G.visited_nodes 
norm_status=(status/G.steps)   
print(status)
print(norm_status)