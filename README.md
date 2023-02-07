# DiffusionNetwork

The following code simulates a diffusion on a Network, performing a random walk and provides an algorithm which allows you to do an agglomerative hierarchical clustering, plot dendograms and plot your network divided into clusters. All the functions needed are in the *networkcluster.py* file. 


# How to perform a Random Walk
A random-walk is the process by which randomly-moving objects wander away from where they started. It describes a path that consists of a succession of random steps on some mathematical space, in our case,
on networks.
1. You have to provide a network **N** and calculate the transition matrix using the function *transition_matrix(**N**)*;
2. You have to decide where the walker will start its walk (**initial_node**);
3. You have to decide the number of steps the walker will do (**steps**);
4. You have to decide the **process_type** : if it is Markovian, you write "*Markov*", if you write something different the code will consider the process non Markovian and so it will set the trasition probability at the current node to 0, so that the walker cannot return where it just was;
5. You have to fix the probability that the walker is absorbed in each node (**abs_prob**);
6. You use the function *simulation(initial_node, steps, process_type, absorption_prob, trans_m)* to get the number of time the walker has been in each node (**visited_nodes**);
7. To repeat this process starting form each node of your network, use the function *random_walk(it, steps, process_type, absorption_prob, trans_m)*. This fucntion repeats also the process **it** times and calculates the average of the outcomes. 

For example, a Random Walk on the Karate Club Graph of the library *NetworkX* following these steps results in: 

```python
import networkx as nx
import sys  
sys.path.insert(0, '/http://localhost:8888/tree/DiffusionNetwork')
from DiffusionNetwork import networkcluster as nc 

N= nx.karate_club_graph()
trans_m = nc.trans_matrix(N)
nc.random_walk(10, 100, "Markov", 0.4, trans_m)
```
```shell
array([[4.3, 0.5, 0.6, ..., 0.2, 0.3, 0.3],
       [1. , 4.6, 0.9, ..., 0. , 0.2, 0.3],
       [0.8, 0.9, 3.8, ..., 0.3, 0.4, 0.8],
       ...,
       [0.4, 0.1, 0.1, ..., 4.7, 0.7, 0.7],
       [0. , 0.2, 0.2, ..., 0.3, 3.6, 1.9],
       [0.2, 0. , 0.2, ..., 0.4, 1.3, 5.3]])
```
