import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

def load_graphs(no_of_graphs, n_hidden, file_loc,start_idx=0, k_days = 366):

    all_df = pd.DataFrame()

    for graph in range(start_idx, start_idx + no_of_graphs):
        # Load data
        loc = os.path.join(file_loc,f'nodes_frames_{graph}.csv')
        nodes = pd.read_csv(loc, sep=';').to_numpy()

        # table : time person_id status
        num_nodes = len(nodes)
        # num_days =len(nodes[0])
        num_days = 120
        matrix = np.zeros((num_nodes*num_days,3))
        for i in range(num_days):
            for j in range(num_nodes):
                matrix[i*num_nodes+j,:] = i,j,nodes[j,i]

        # Define the feature vector as a bag of word
        # integer encode
        label_encoder = LabelEncoder()
        label_encoder.fit([0,1,2,3,4])
        integer_encoded = label_encoder.transform(matrix[:,2])
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(np.array([[0],[1],[2],[3],[4]]))
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.transform(integer_encoded)

        # create data structure
        df=pd.DataFrame(np.concatenate(([matrix[:,0:2], onehot_encoded, matrix[:,2].reshape((len(matrix[:,2]),1))]),axis=1))
        df.columns=['t','id','x1', 'x2', 'x3', 'x4', 'x5', 'y']
        total_nodes = len(df['id'])
        #state:'1' the node state is known, '0' unknown
        df['state'] = np.ones(df.shape[0])

        # for each person
        for i in range(num_nodes):
            # select sub-matrices [x1, ..., x5] with same id along all days 't'
            tmp_mat = df.values[np.arange(i,total_nodes,num_nodes),2:-2]
            tmp_mat_kdays = df.values[np.arange(i,total_nodes,num_nodes),2:-2]
            # and integrate them along days
            for j in range(1,len(tmp_mat)):
                if j >= k_days:
                    # now move tmp_mat as new x1...x
                    tmp_mat[j,:] += tmp_mat[j-1,:]
                    tmp_mat_kdays[j,:] = tmp_mat[j,:] - tmp_mat[j-k_days,:]
                else:
                    tmp_mat[j,:] += tmp_mat[j-1,:]
                    tmp_mat_kdays[j,:] += tmp_mat_kdays[j-1,:]

            df.values[np.arange(i,total_nodes,num_nodes),2:-2] = tmp_mat_kdays

        hidden_nodes = n_hidden #%
        unknown_indexes = random.sample(range(0, num_nodes), int(hidden_nodes*num_nodes))
        unknown_indexes.sort()
        tmp_values = []
        # mask this 20% of the nodes along all days
        for i in unknown_indexes:
            #print(df.values[np.arange(i,total_nodes,num_nodes),2:-1])
            tmp_values.append(df.values[np.arange(i,total_nodes,num_nodes),2:-2])
            df.values[np.arange(i,total_nodes,num_nodes),2:-2] = 0
            #indicate the state of unknown nodes
            df.values[np.arange(i,total_nodes,num_nodes),-1] = 0

        all_df = all_df.append(df, ignore_index=True)


    return all_df, num_nodes, num_days