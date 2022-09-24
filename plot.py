import sys
import networkx as nx
from CSom import *
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt


RADIUS_MAP=16
def load_data(filepath):
    with open(filepath, 'rb') as inp:
        data=pickle.load(inp)
    return data


print("Loading...")

processed_data=load_data('./data_preprocess')

with open('./ksom.nx') as inp:
    loaded_graph=pickle.load(inp)

print("Finish loading")
labels_node={}

for i in range(0, 256):
    labels_node[i]=2.0

for i in range(0, len(processed_data)):
    labels_node[RADIUS_MAP*RADIUS_MAP+i]=float(processed_data[i][1]==True)


values=[labels_node.get(val, 5.0) for val in loaded_graph.nodes()]
nx.draw_spring(loaded_graph, with_labels = False, cmap=plt.get_cmap('viridis'), node_color=values)
# nx.draw_networkx_edge_labels(G,edge_labels=labels)

plt.show()