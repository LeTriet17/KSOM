import sys
import networkx as nx
from CSom import *
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

def load_model(filepath):
    with open(filepath, 'rb') as inp:
        model=pickle.load(inp)
    return model

def load_data(filepath):
    with open(filepath, 'rb') as inp:
        data=pickle.load(inp)
    return data


model=load_model('./ksom.ckpt')
processed_data=load_data('./data_preprocess')

RADIUS_MAP=16

def getDistanceNode2Node(a,b):
    a=np.squeeze(np.asarray(a))
    b=np.squeeze(np.asarray(b))
    temp=a-b
    sum_sq = np.dot(temp.T, temp)
    return math.sqrt(sum_sq)




def build_graph_ksom():
    graph= nx.Graph()

    edge_dict={}
    radius=1
    edge_weighted_list=[]
    pos={}
    for ix, iy in np.ndindex(model.m_Som.shape):
        idx_cnode=RADIUS_MAP*ix+iy
        pos[idx_cnode]=(ix, iy)
        for i in range (-radius,radius+1):
            for j in range (0, radius+1):
                if (i==0 and j==0):
                    continue
                x_idx, y_idx= ix+i, iy+j
                if (0<=x_idx and x_idx<RADIUS_MAP and 0<=y_idx and y_idx<RADIUS_MAP):
                    # print(f"CNode in {ix}, {iy} near to {x_idx}, {y_idx}")
                    idx_cnode_neighbor=RADIUS_MAP*x_idx+y_idx
                    if (idx_cnode,idx_cnode_neighbor) not in edge_dict and (idx_cnode_neighbor,idx_cnode) not in edge_dict:
                        # print("Go here")
                        weights=model.m_Som[(ix, iy)].CalculateDistance2CNode(model.m_Som[x_idx, y_idx])
                        # print(weights)
                        
                        edge_weighted_list.append([idx_cnode, idx_cnode_neighbor, round(weights,2)])
                        edge_dict[(idx_cnode, idx_cnode_neighbor)]=1
    # graph.add_weighted_edges_from(edge_weighted_list)
    return graph, pos, edge_weighted_list


def build_leaf_node_ksom(lst_weight_edge):
    global G
    idx_start=RADIUS_MAP*RADIUS_MAP
    
    corpus=[x[0] for x in processed_data]
    PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    PNodes = PNodes.fit_transform(corpus).todense()
    sum_quan_err=0
    for i in range(PNodes.shape[0]):
      
        SuitNode, ix, iy = model.FindBestMatchingNode(PNodes[i])
        weight_val=math.sqrt(SuitNode.CalculateDistance(np.squeeze(np.asarray(PNodes[i]))))      
        sum_quan_err+=weight_val
        lst_weight_edge.append([RADIUS_MAP*iy+ix, idx_start, math.sqrt(SuitNode.CalculateDistance(np.squeeze(np.asarray(PNodes[i]))))])
        for node_idx in model.m_Som[iy, ix].PNodes:
            weight_pnode2pnode=getDistanceNode2Node(model.m_Som[iy, ix].PNodes[node_idx].vector, PNodes[i])
            lst_weight_edge.append([node_idx, idx_start, weight_pnode2pnode])
        SuitNode.addPNode(corpus[i], PNodes[i], idx_start)
        idx_start+=1
    print(f"Quantization Error {sum_quan_err/len(corpus)}")
    return lst_weight_edge


G, pos, edge_list=build_graph_ksom()
edge_list=build_leaf_node_ksom(edge_list)
G.add_weighted_edges_from(edge_list)


labels = nx.get_edge_attributes(G,'weight')
# print(labels)

with open('ksom.nx', 'wb') as inp:
    data=pickle.dump(G, inp)
labels_node={}

for i in range(0, 256):
    labels_node[i]=2.0

for i in range(0, len(processed_data)):
    labels_node[RADIUS_MAP*RADIUS_MAP+i]=float(processed_data[i][1]==True)
values=[labels_node.get(val, 5.0) for val in G.nodes()]


nx.draw(G, with_labels = False, cmap=plt.get_cmap('viridis'), node_color=values)
# nx.draw_networkx_edge_labels(G,edge_labels=labels, pos=pos)

plt.show()