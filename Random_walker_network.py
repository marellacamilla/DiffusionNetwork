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
        
         
         self.last_visited_node = initial_node
         self.steps += 1
         self.visited_nodes[initial_node] += 1
         return initial_node
    
     def sim_step(self, current_node):
         """
         For each step of the simulation:
 
        
         If the process is markovian (not_markov = False) the simulation 
         can proceed to any node.
         The next_node is chosen randomly in the list of the nodes of the graph, 
         the weight is the row of the transition matrix corresepondent to the current_node.
         The number of steps of the simulation is incremented by one.
         The number of times the walker has been in the node = next_node is incremented by one.
         
         If the process is not Markovian (not_markov = True) the walker can go only on states different from  the last_visited_node.
         The porbability of transition to the last_visited_node is set as zero. Then the procedure is the same as described in the previous case.

         Parameters
         ----------
         current_node : int
             The node where the walker is.

         Returns
         -------
         next_node : int
             The next node the walker goes to.
             
             
        

         """
         
         p=np.array(self.trans_m[current_node, :].todense())[0]
         
         #if np.sum(p)==0: #check if p == 0
         #   return self.initialize(current_node)
         
          
         if self.not_markov:
            p=np.array(self.trans_m[current_node, :].todense())[0]
            p[self.last_visited_node]=0
            
            next_node=np.random.choice(np.arange(0, self.n_nodes), p=p)
            return next_node
            
            
         else:
             
             next_node =random.choices(np.arange(0, self.n_nodes), weights=p, k=1)
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

     
        



    
