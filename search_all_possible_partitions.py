
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import networkx as nx


# In[15]:

import time
import sys


# In[7]:

from pyinfomap import load_and_process_graph, Clustering


# In[8]:

def load_graph(graph_fname="2009_figure3ab.net"):
    with open(graph_fname, 'r') as f:
        graph = load_and_process_graph(f)
    return graph
graph = load_graph()


# In[9]:

def get_mdl(graph, modules):
    clustering = Clustering(graph, modules)
    return clustering.get_mdl()


# In[10]:

from sympy.utilities.iterables import multiset_partitions


# In[11]:

modules = [[x] for x in graph.nodes()]


# In[19]:

# initial mdl
initial_mdl = get_mdl(graph, modules)
print("initial MDL: {}".format(initial_mdl))


# In[20]:

# %%time
best_mdl = initial_mdl
best_partition = modules
best_i = 0
i = 0
start = time.time()
print("Looping through all partitions for the {}-node network".format(graph.number_of_nodes()))
for p in multiset_partitions(graph.nodes()):
    mdl = get_mdl(graph, p)
    if mdl < best_mdl:
        best_mdl = mdl
        best_partition = p
        best_i = i
    i += 1
    if i in [1000, 10000, 1e5, 1e6, 1e7] or i % 5e7 == 0:
        print("{} partitions tried. {:.1f} seconds so far. Best: {} (iter. # {})".format(i, time.time()-start, best_mdl, best_i))
        sys.stdout.flush()
print("")
print("Done. {} partitions tried.".format(i))
print("Best MDL: {} (found on iteration {})".format(best_mdl, best_i))
print(best_partition)


# In[ ]:



