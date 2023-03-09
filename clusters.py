# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:23:01 2023

@author: HP
"""
import numpy as np
import chain 
import networkcluster as nc
import sys
from sys import argv
import configparser
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import networkx as nx


config = configparser.ConfigParser()
config.read(sys.argv[1])

t = config.get('settings', 't')
t=int(t)

destination1 = config.get('paths','dendo_pic')
destination2 = config.get('paths','net_pic')

N=chain.G
trans_m=nc.trans_matrix(N)

#Load the outcomes obatined from the simulation of the random walk
with open('random_walk_data.npy', 'rb') as f:
    data = np.load(f)

#clustering
X = nc.clustering(t, nc.linkage_matrix(data))
print("The number of clusters is:", max(X), ".")
print(X)

for i in range(1, max(X)+1):
    count = np.count_nonzero(X == i)
    print("The cluster", i , "is composed by", count, "nodes.")

#Plots
def DendoPlot():
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(nc.linkage_matrix(data), show_leaf_counts=True)
    plt.axhline(y = t, color = 'r', linestyle = '-')
    plt.show()
    fig.savefig(destination1)
    
def NetPlot():
    color=('red', 'blue', 'yellow', 'green', 'purple', 'pink', 'white')
    
    k=max(X)
    color_map=[]
    for s in range(1,k+1):
         for i in range(trans_m.shape[0]):
             if X[i]==s:
                 color_map.append(color[s])
             continue
         

    layout_pos = nx.spring_layout(N) 
    fig = plt.figure(figsize=(7, 5))       
    nx.draw(N, pos=layout_pos, node_color=color_map, with_labels=True)
    plt.show()
    fig.savefig(destination2)
    
    
    
DendoPlot()
NetPlot()

print("Plots saved!")