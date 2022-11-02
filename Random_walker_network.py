import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt

class Markov_Graph:
     """ Class which identifies a random walk on a Markov (or Not Markov) graph """
     
     def __init__(self, transition_matrix, absorption, not_markov, starting_p):
        
        """ 
        
        
        Parameters: 
            
        -------
        transition_matrix: array
            The matrix of the probabilities of transition between each pair of 
            nodes
            
        absorption: array
          The probability that the walker will be absorbed for each node
            
        not_markov: bool
            if not_markov = True, the process is not Markovian so the walker 
            can't return back to the previus node
            
        starting_p : list
           The starting probability of the simulation
            
        n_nodes : int
            The number of nodes = the number of rows of the transition matrix
            
        visited_nodes : array
            The number of times the walker has been in each node
            
        steps : int
            The number of steps the walker has made 
            
            
        """
        self.trans_m = transition_matrix
        self.n_nodes = self.trans_m.shape[0] 
        self.absorption = absorption 
        self.last_visited_node = None 
        self.visited_nodes = np.array([0 for i in range(self.n_nodes)]) 
        self.steps = 0 
        self.not_markov = not_markov
        self.starting_p = starting_p
        
     def initialize(self, initial_node) : 
         """
         The function returns the starting point of the simulation,
         the first node where we consider the walker.
         The number of step of the simulation is incremented by one and also the 
         correspondent value in the arrayof the visited_nodes.

         Returns
         -------
         initial_node : int
             
         

         """
        
         #initial_node = np.random.choice(np.arange(0, self.n_nodes), p=self.starting_p)
         
         self.last_visited_node = initial_node
         self.steps += 1
         self.visited_nodes[initial_node] += 1
         return initial_node
    
     def sim_step(self, current_node):
         """
         For each step of the simulation:
         
         If the process is not Markovian (not_markov = True) the walker can go only on states different from  the last_visited_node.
        
        
         If the process is markovian (not_markov = False) the simulation 
         can proceed to any node.
         The next_node is chosen randomly in the list of the nodes of the graph.
         The number of steps of the simulation is incremented by one.
         The number of times the walker has been in the node = next_node is incremented by one

         Parameters
         ----------
         current_node : int
             The node where the walker is.

         Returns
         -------
         next_node : int
             The next node the walker goes to.
             
             
        

         """
         

         if self.not_markov:
            
            p=np.array(self.trans_m[current_node])
            
            p[self.last_visited_node]=0
            
            #next_node = np.random.choice(np.arange(0, self.n_nodes), p=p)
            
            
            #list of all the nodes
            #list_nodes=np.arange(0, self.n_nodes) 
            
            #list of nodes allowed: we have excluded the last_visited node
            #allw_nodes=np.delete(list_nodes, self.last_visited_node)
            
            #Transition probabilities of the nodes allowed
            q=np.delete(p[self.last_visited_node], 0) 
            #We have added to the probabilities of each nodes the probability of the last_visited_node normalized by the number of nodes available
            #r=q+[p[self.last_visited_node]/(self.n_nodes-1), p[self.last_visited_node]/(self.n_nodes-1)]
            
            next_node=np.random.choice(np.arange(0, self.n_nodes), p=self.trans_m[current_node])
            
            return next_node
            
            
         else:
            next_node = np.random.choice(np.arange(0, self.n_nodes), p=self.trans_m[current_node,:])
            self.last_visited_node = next_node
            self.steps += 1
            self.visited_nodes[next_node] += 1
            return next_node
        
     def absorbed(self, current_node):
         """
         The function evaluates the probability of absorption at the current_node.
         It returns True if the walker is absorbed, False if it is not.

         Parameters
         ----------
         current_node : int
             The node where the walker is.

         Returns
         -------
         bool
             
          

         """
         p=self.absorption[current_node]
         
         prob_absorb = np.random.choice([1,0], 1, [p, 1-p])
         
         if prob_absorb == 1:
             return 1
         return 0

     
        



    
