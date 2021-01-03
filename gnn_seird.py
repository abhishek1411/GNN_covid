# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:55:54 2020

@author: Matteo
"""

# from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import random
import time
import math
random.seed(1)
import os
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric import utils
from torch_geometric.nn import GCNConv,SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


from gnn_seird_dataloader import load_graphs
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

torch.cuda.empty_cache()

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

######## CLASS

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, 5)
        
    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv1(x, edge_index)
        x = x.relu() # the classical activation function
        x = F.dropout(x, p=0.25, training=self.training)# and dropout to avoid overfitting
        
        # Second Message Passing layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        
        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(x), dim=1) #is the classification layer
        return x

######## FUNCTIONS

def train(_train_data):
    loss_all = 0
    total_graphs = 0
    for k, _data in enumerate(_train_data):
        #Uncomment it if using Inmemory data
        _data = _data.to(device)
        model.train() # is the function of the parent class
        optimizer.zero_grad() # Reset gradients
        # use all data as input, because all nodes have node features
        out = model(_data)
        loss = criterion(out, _data.y)
        loss.backward()
        optimizer.step()
        loss_all += _data.num_graphs * loss.item()
        total_graphs += _data.num_graphs
    return loss_all / total_graphs


def test(_test_data):
    test_correct = 0
    test_nodes = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for k, _data in enumerate(_test_data):
            _data = _data.to(device)
            model.eval() # is the parent function
            out = model(_data)
            # use the classes with highest probability
            pred = out.argmax(dim=1)
            node_state = np.hstack(_data.state)
            unknown_indexes = np.where(node_state == 0)[0]
            pred_correct = (pred[unknown_indexes] == _data.y[unknown_indexes])
            test_correct += pred_correct.sum()
            test_nodes += unknown_indexes.shape[0]
            y_true = np.hstack((y_true, _data.y[unknown_indexes].cpu().detach().numpy()))
            y_pred = np.hstack((y_pred, pred[unknown_indexes].cpu().detach().numpy()))

    cm = confusion_matrix(y_true, y_pred)  # (y_true,y_prediction)
    test_acc = int(test_correct) / test_nodes
    return test_acc,cm

def load_data(path, data_frame):
    ###change the data loader
    x = torch.tensor(data_frame.values[:,2:-2], dtype=torch.float)
    y = torch.tensor(data_frame['y'], dtype=torch.long)
    data_list = []
    for j in range(no_graphs):
        loc = os.path.join(path,f'Adjacency_matrix_edgelist_{j}.csv')
        edges = pd.read_csv(loc, header=None, sep=';').to_numpy()
        # Edges list: in the format (head, tail); the order is irrelevant
        edge_index = utils.to_undirected(torch.tensor(edges, dtype=torch.long).t())
        for i in range(num_days):
            start_idx = i*num_nodes + j*num_nodes*num_days
            end_idx = i*num_nodes+ j*num_nodes*num_days + num_nodes
            data_list.append(Data(x=x[start_idx:end_idx,:], y=y[start_idx:end_idx],
                                  edge_index=edge_index.to(device), state=data_frame.values[start_idx:end_idx,-1]))

    return data_list
######## PREPROCESS DATA

class GNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNNDataset, self).__init__(root,transform, pre_transform)
        self.df = df
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../data/chk.dataset']

    def download(self):
        pass

    def process(self):
        ####cHANGE THIS TO ACCEPT BOTH TEST AND TRAIN PATHS:df and path
        path = './graphs'
        data_list = []
        x = torch.tensor(df.values[:,2:-2], dtype=torch.float)
        y = torch.tensor(df['y'], dtype=torch.long)
        for j in range(no_graphs):
            loc = os.path.join(path,f'Adjacency_matrix_edgelist_{j}.csv')
            edges = pd.read_csv(loc, header=None, sep=';').to_numpy()
            # Edges list: in the format (head, tail); the order is irrelevant
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            for i in range(num_days):
                start_idx = i*num_nodes + j*num_nodes*num_days
                end_idx = i*num_nodes+ j*num_nodes*num_days + num_nodes
                data_list.append(Data(x=x[start_idx:end_idx,:], y=y[start_idx:end_idx],
                                      edge_index=edge_index, state=df.values[start_idx:end_idx,-1]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Load data
no_graphs = 1
no_graphs_test = 1
hidden_nodes = 0.5
k_days = 15
print(f'Train graphs:{no_graphs},Test graphs:{no_graphs_test},'
      f'hidden_nodes:{hidden_nodes}, k_days:{k_days}')
file_loc_train = './graphs'
file_loc_test = './graphs_test'
# num_nodes = 1000
# num_days =365
start_idx_graph = 0
df, num_nodes, num_days = load_graphs(no_graphs, hidden_nodes, file_loc_train, start_idx_graph,k_days)
print(f'No.of_days_used:{num_days}')
# feature scaling
sc = StandardScaler()
known_idx_train = np.where(df['state'] == 1)[0]
scaler = sc.fit(df.values[known_idx_train,2:-2])
df.values[known_idx_train,2:-2] = scaler.transform(df.values[known_idx_train,2:-2])

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train_data_list = GNNDataset('./')
# train_data_list = train_data_list.shuffle()
# train_dataset = DataLoader(train_data_list, batch_size=512)

train_data_list = load_data(file_loc_train, df)
num_features = train_data_list[-1].num_features
train_indices = [id for id in range(len(train_data_list))]
random.shuffle(train_indices)
train_dataset = DataLoader(train_data_list, batch_size=512, sampler=train_indices)

num_features = 5
# train_dataset.num_classes = 5 # S,E,I,R,D

# Initialize Model
model = GCN(hidden_channels=16)
print(model)
model = model.to(device)

learning_rate = 0.005 # step for gradient descendent method for learning (?)
decay = 5e-4 #decay of importance of learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# Define loss function (CrossEntropyLoss for Classification problems with probability distribution)
w0 = np.where(df['y'] == 0)[0].shape[0]
w1 = np.where(df['y'] == 1)[0].shape[0]
w2 = np.where(df['y'] == 2)[0].shape[0]
w3 = np.where(df['y'] == 3)[0].shape[0]
w4 = np.where(df['y'] == 4)[0].shape[0]
max_w = max(w0,w1,w2,w3,w4)
weight = torch.tensor([max_w/w0,max_w/w1,max_w/w2,max_w/w3,max_w/w4*0.5]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weight)
# cross entropy compare probabilities, and we have probabilities because of softmax

# Train !
del df
del train_data_list
losses = []
start = time.time()
for epoch in range(10):
    loss = train(train_dataset)
    losses.append(loss)
    if epoch % 100 == 0:
        # train_acc,_ = test(train_dataset)
        # test_acc,_ = test(test_dataset)
        print(f'[{time_since(start)}], Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Train results
# import seaborn as sns
# losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
# loss_indices = [i for i,l in enumerate(losses_float)]
# sns.lineplot(loss_indices, losses_float)
# plt.show()


######################################################
# Test the model with different unknown nodes

del train_dataset

for n_test in range(3):
    torch.cuda.empty_cache()
    start_idx_graph = n_test * no_graphs_test
    df_test, _, _ = load_graphs(no_graphs_test, hidden_nodes, file_loc_test,start_idx_graph, k_days)
    known_idx_test = np.where(df_test['state'] == 1)[0]
    df_test.values[known_idx_test,2:-2] = scaler.transform(df_test.values[known_idx_test,2:-2])
    test_data_list = load_data(file_loc_test,df_test)
    test_indices = [id for id in range(len(test_data_list))]
    test_dataset = DataLoader(test_data_list, batch_size=256, sampler=test_indices)
    test_acc, conf_matrix = test(test_dataset)
    print(f'Test_set:{n_test}, Test Accuracy: {test_acc:.4f}, confusion_matrix:\n'
          f'{conf_matrix}')
    print(f'Accuracy_infected = {conf_matrix[2,2]/np.sum(conf_matrix[2,:])}\n'
          f'Precision_infected = {conf_matrix[2,2]/np.sum(conf_matrix[:,2])}')
    del df_test
    del test_data_list
    del test_indices
    del test_dataset

# Improving the model
# Cross-Validation
# Hyperparameter Optimization
# Different layer types GCN, GAT...
# Different message passing layers
# Including edge features
# Featire scaling

#networkx to visulize the graph
#is addself loop required in forward?I think so it is done
# GCN with skip connections