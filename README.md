# DiffusionNetwork

The aim of this project is to simulate a diffusion process on a Network, through a random walk. It contains also an algorithm which allows to do an agglomerative hierarchical clustering of the data, plot dendograms and plot your network divided into clusters.
All the functions needed are in the *networkcluster.py* file. 

# Installation

To install the application clone the repository DiffusionNetwork:
```python
git clone https://github.com/marellacamilla/DiffusionNetwork
cd DiffusionNetwork
```

In order to install the library some other libraries are necessary, so we can install them with:
```python
pip install -r requirements.txt
```
Running:
```python
!pytest test_networkcluster.py
```

will execute all test functions inside the test_networkcluster.py file. 
The library was built and tested with python version 3.9.12.

# How to perform a Random Walk
A random-walk is the process by which randomly-moving objects wander away from where they started. It describes a path that consists of a succession of random steps on some mathematical space, in our case,
on networks.
1. You have to provide a network **N** and calculate the transition matrix using the function *transition_matrix(**N**)*;
2. You have to decide where the walker will start its walk (**initial_node**);
3. You have to decide the number of steps the walker will do (**steps**);
4. You have to decide the **process_type** : if it is Markovian, you write "*Markov*", if you write something different the code will consider the process non Markovian and  it will set the trasition probability at the current node to 0, so that the walker cannot return where it just was;
5. You have to fix the probability that the walker is absorbed in each node (**abs_prob**);
6. You use the function *simulation(initial_node, steps, process_type, absorption_prob, trans_m)* to get the number of time the walker has been in each node (**visited_nodes**);
7. To repeat this process starting from each node of your network, use the function *random_walk(it, steps, process_type, absorption_prob, trans_m)*. This fucntion repeats also the process **it** times and calculates the average of the outcomes. 

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
```python
Out[7]:
array([[4.3, 0.5, 0.6, ..., 0.2, 0.3, 0.3],
       [1. , 4.6, 0.9, ..., 0. , 0.2, 0.3],
       [0.8, 0.9, 3.8, ..., 0.3, 0.4, 0.8],
       ...,
       [0.4, 0.1, 0.1, ..., 4.7, 0.7, 0.7],
       [0. , 0.2, 0.2, ..., 0.3, 3.6, 1.9],
       [0.2, 0. , 0.2, ..., 0.4, 1.3, 5.3]])
```
# How to do clustering
The algorithm proposed allows to:
* Compute the linkage matrix (**link_m**), with *linkage_matrix(**data**)* function, where as **data** you have to insert the output of the *random_walk* function (like Out[7]); 
* Represent the agglomerative hierarchical clustering with a dendogram ( *plot_dendogram(**t**, **link_m**)*), drawing an horizontal line at hegiht **t** to form then the flat clusters and determine the number of clusters; 
```python
data = Out[7]
link_m = nc.linkage_matrix(data)
nc.plot_dendogram(11, link_m) # the distance threshold at which the dendogram is cut is 11
```
![dendogram](/readme_images/dendo.png)

* form flat clusters, according to the threshold **t**, with the function *clustering(**t**, **link_m**)*. For example, you can print the cluster to which each node belongs and determine the number of clusters formed;
```python
clusters = nc.clustering(11, link_m)
n = max(clusters)

print(clusters)
print(n)
```
```python
[2 2 2 2 2 2 2 2 1 1 2 2 2 2 1 1 2 2 1 2 1 2 1 1 1 1 1 1 1 1 1 1 1 1]
2
```
* visualize the partition into clusters on the network, with the function *plot_clustering(**N**,**t**, **X** ,**trans_m**)*.
```python
nc.plot_clustering(N, 11, clusters, trans_m)

```
![network](/readme_images/net.png)
