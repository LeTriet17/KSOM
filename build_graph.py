import sys
import networkx as nx
from CSom import *
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from utils import *





model=load_model('../ksoms.ckpt')
processed_data=load_data('../data_preprocess')
PNodes_arr=create_pnode(model, processed_data)

RADIUS_MAP=16


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
                        weights=model.m_Som[(ix, iy)].CalculateDistance2CNode(model.m_Som[x_idx, y_idx], [0.5, 0.5])
                        # print(weights)
                        
                        edge_weighted_list.append([idx_cnode, idx_cnode_neighbor, round(weights,2)])
                        edge_dict[(idx_cnode, idx_cnode_neighbor)]=1
    # graph.add_weighted_edges_from(edge_weighted_list)
    return graph, pos, edge_weighted_list


def build_leaf_node_ksom(lst_weight_edge):
    global G
    idx_start=RADIUS_MAP*RADIUS_MAP
    sum_quan_err=0
    for i in range(PNodes_arr.shape[0]):
      
        SuitNode, ix, iy = model.FindBestMatchingNode(PNodes_arr[i])
        weight_val= model.CalculateDistance_PNode2CNode(PNodes_arr[i], SuitNode)
        sum_quan_err+=weight_val
        lst_weight_edge.append([RADIUS_MAP*iy+ix, idx_start, weight_val])
        for node_idx in model.m_Som[iy, ix].PNodes:
            weight_pnode2pnode=model.calcdistance2PNode(model.m_Som[iy, ix].PNodes[node_idx], PNodes_arr[i])
            lst_weight_edge.append([node_idx, idx_start, weight_pnode2pnode])
        SuitNode.addPNode(PNodes_arr[i], idx_start)
        idx_start+=1
    # print(f"Quantization Error {sum_quan_err/len()}")
    return lst_weight_edge


G, pos, edge_list=build_graph_ksom()
edge_list=build_leaf_node_ksom(edge_list)
G.add_weighted_edges_from(edge_list)


# labels = nx.get_edge_attributes(G,'weight')
# print(labels)

with open('ksom.nx', 'wb') as inp:
    data=pickle.dump(G, inp)
labels_node={}

for i in range(0, 256):
    labels_node[i]=1.0

for i in range(0, len(processed_data)):
    labels_node[RADIUS_MAP*RADIUS_MAP+i]=float(processed_data[i][-1]==True)
values=[labels_node.get(val, 5.0) for val in G.nodes()]


import torch

adj = nx.to_scipy_sparse_array(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)

labels=np.array(values).astype(np.int64)
embeddings=[0]*(16*16+len(processed_data))
for ix, iy in np.ndindex(model.m_Som.shape):
    temp=tuple()
    for w in model.m_Som[ix, iy].dWeights:
        temp+=(w,)
        embeddings[16*ix+iy]= np.concatenate(temp, axis=None)
corpus=[x[0] for x in processed_data]
PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
PNodes = PNodes.fit_transform(corpus).todense()
for i in range (0, PNodes.shape[0]):
    embeddings[16*16+i]=np.squeeze(np.asarray(PNodes[i]))
for i in range(PNodes_arr.shape[0]):
    temp = tuple()
    for j in range(2):
        temp+=(PNodes_arr[i].getvector(j),)
    embeddings[16*16+i]=np.concatenate(temp, axis=None)
print(embeddings[16*16+1].shape)
embeddings=np.array(embeddings)
# nx.draw(G, with_labels = False, cmap=plt.get_cmap('viridis'), node_color=values)
# # nx.draw_networkx_edge_labels(G,edge_labels=labels, pos=pos)

# plt.show()

import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
class FakeNewsDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(FakeNewsDataset, self).__init__('.', transform, None, None)

        data = Data(edge_index=edge_index)
        
        data.num_nodes = G.number_of_nodes()
        
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32)
        
        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        
        data.num_classes = 2

        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())), 
                                                            pd.Series(labels),
                                                            test_size=0.4, 
                                                            random_state=42)
        
        n_nodes = G.number_of_nodes()
        print(X_train.index)
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        for i in range (0, 256):
          train_mask[i]=False
        test_mask[X_test.index] = True
        for i in range (0, 256):
          test_mask[i]=True

        data['train_mask'] = train_mask
        data['test_mask'] = test_mask

        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
dataset = FakeNewsDataset()
data = dataset[0]

print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(50*'=')

# There is only one graph in the dataset, use it as new data object
data = dataset[0]  

# Gather some statistics about the graph.
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Is undirected: {data.is_undirected()}')

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(dataset.num_features, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.out = nn.Linear(hidden_channels_2, dataset.num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x,p=0.25, training=self.training)

        # Output layer 
        x = F.log_softmax(self.out(x), dim=1)
        return x

model = GCN(hidden_channels_1=256, hidden_channels_2=256)
print(model)
# Use GPU
print("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)
      # Check against ground-truth labels.

      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      # Derive ratio of correct predictions.
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
      return test_acc, train_acc
      

losses = []
for epoch in range(0, 1000):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

sample = 400
print(model(data.x, data.edge_index).shape)
pred = model(data.x, data.edge_index)

TP =0
FP = 0
FN = 0
TN = 0

for i in range (256, 2486):
  pred_labels=np.argmax(pred[i].detach().cpu().numpy())
  if data.test_mask[i]==True:
    if data.y[i]==1 and pred_labels==1:
      TP+=1
    elif data.y[i]==1 and pred_labels==0:
      FN+=1
    elif data.y[i]==0 and pred_labels==0:
      TN+=1
    else:
      FP+=1
print(TP, FP, FN, TN)
print(f"Accuracy {(TP+TN)/(FP+FN+TP+TN)}")
print(f"Precision {(TP)/(TP+FP)}")
print(f"Recall {(TP)/(TP+FN)}")
