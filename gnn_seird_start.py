# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:55:54 2020

@author: Matteo
"""

from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import random

random.seed(1)

import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv

from sklearn.preprocessing import OneHotEncoder, StandardScaler

######## CLASS

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = GCNConv(dataset.dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, 5)
        
    def forward(self, x, edge_index):
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv1(x, edge_index)
        x = x.relu() # the classical activation function
        x = F.dropout(x, p=0.5, training=self.training)# and dropout to avoid overfitting
        
        # Second Message Passing layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(x), dim=1) #is the classification layer
        return x

######## FUNCTIONS

def train():
    model.train() # is the function of the parent class
    optimizer.zero_grad() # Reset gradients
    # use all data as input, because all nodes have node features
    out = model(dataset.dataset.x, dataset.dataset.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval() # is the parent function
    out = model(dataset.dataset.x, dataset.dataset.edge_index)
    # use the classes with highest probability
    pred = out.argmax(dim=1)
    # check against ground-truth labels
    test_correct = (pred[unknown_indexes] == dataset.dataset.y[unknown_indexes])
    #test_correct = (pred == dataset.dataset.y)
    # derive ratio of correct predictions
    test_acc = int(test_correct.sum()) / len(unknown_indexes)
    #test_acc = int(test_correct.sum()) / total_nodes
    return test_acc

######## PREPROCESS DATA

# Load data
nodes = pd.read_csv('./nodes_frames.csv', sep=';').to_numpy()
edges = pd.read_csv('./Adjacency_matrix_edgelist.csv', header=None, sep=';').to_numpy()

# table : time person_id status
num_nodes = len(nodes)
num_days =len(nodes[0])
matrix = np.zeros((num_nodes*num_days,3))
for i in range(num_days):
    for j in range(num_nodes):
        matrix[i*num_nodes+j,:] = i,j,nodes[j,i]

# Define the feature vector as a bag of word
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(matrix[:,2])
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# create data structure
df=pd.DataFrame(np.concatenate(([matrix[:,0:2], onehot_encoded, matrix[:,2].reshape((len(matrix[:,2]),1))]),axis=1))
df.columns=['t','id','x1', 'x2', 'x3', 'x4', 'x5', 'y']
total_nodes = len(df['id'])

# for each person
for i in range(num_nodes):
    # select sub-matrices [x1, ..., x5] with same id along all days 't'
    tmp_mat = df.values[np.arange(i,total_nodes,num_nodes),2:-1]
    # and integrate them along days
    for j in range(1,len(tmp_mat)):
        tmp_mat[j,:] += tmp_mat[j-1,:]
    # now move tmp_mat as new x1...x
    df.values[np.arange(i,total_nodes,num_nodes),2:-1] = tmp_mat

# featire scaling
sc = StandardScaler()
df.values[:,2:-1] = sc.fit_transform(df.values[:,2:-1])

# take 20% of nodes as unknown status
hidden_nodes = 0.2 #%
unknown_indexes = random.sample(range(0, num_nodes), int(hidden_nodes*num_nodes))
unknown_indexes.sort()
tmp_values = []
# mask this 20% of the nodes along all days
for i in unknown_indexes:
    #print(df.values[np.arange(i,total_nodes,num_nodes),2:-1])
    tmp_values.append(df.values[np.arange(i,total_nodes,num_nodes),2:-1])
    df.values[np.arange(i,total_nodes,num_nodes),2:-1]=0

# The embedding for each node is the vector of vector [x1, ..., x5]
#np.random.shuffle(df.values)
#df.sample(frac=1) # shuffle rows
x = torch.tensor(df.values[:,2:-1], dtype=torch.float)

# Define the status of each node
y = torch.tensor(df['y'], dtype=torch.long)
# Edges list: in the format (head, tail); the order is irrelevant
edge_index = torch.tensor(edges, dtype=torch.long).t()

# Putting them together, we can create a Data object
data = Data(x=x, y=y, edge_index=edge_index)

# Create dataset with batch size as a snapshot of 1 day
dataset = DataLoader(data, batch_size=num_nodes, shuffle=True)
dataset.num_classes = 5 # S,E,I,R,D
# there are as many graph as the number of iterations : thus, 500 graphs!

model = GCN(hidden_channels=16)
print(model)

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Initialize optimizer
learning_rate = 0.01 # step for gradient descendent method for learning (?)
decay = 5e-4 #decay of importance of learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

# Define loss function (CrossEntropyLoss for Classification problems with probability distribution)
criterion = torch.nn.CrossEntropyLoss()
# cross entropy compare probabilities, and we have probabilities because of softmax

# Train !
losses = []
for epoch in range(1000):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Train results
import seaborn as sns
losses_float = [float(loss.cpu().detach().numpy()) for loss in losses] 
loss_indices = [i for i,l in enumerate(losses_float)] 
plt = sns.lineplot(loss_indices, losses_float)
plt


######################################################

# Test the model with different unknown nodes

# first refill the old unknown by trasforming them in known
count = 0
for i in unknown_indexes:
    dataset.dataset.x[np.arange(i,total_nodes,num_nodes), :] = torch.from_numpy(tmp_values[count]).to(dataset.dataset.x)
    count = count + 1

# now find the people to mask for testing the GNN
unknown_indexes = random.sample(range(0, num_nodes), int(hidden_nodes*num_nodes))
unknown_indexes.sort()
for i in unknown_indexes:
    dataset.dataset.x[np.arange(i,total_nodes,num_nodes), :] = torch.from_numpy(np.zeros(5)).to(dataset.dataset.x)

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

# Improving the model
# Cross-Validation
# Hyperparameter Optimization
# Different layer types GCN, GAT...
# Different message passing layers
# Including edge features
# Featire scaling