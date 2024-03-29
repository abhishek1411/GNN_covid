import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

class GNNDataset(InMemoryDataset):
    def __init__(self, root,df, transform=None, pre_transform=None):
        super(GNNDataset, self).__init__(root,transform, pre_transform)
        self.df = df
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./data/chk.dataset']

    def download(self):
        pass

    def process(self):

        data_list = []

        # process by session_id
        grouped = self.df.groupby('t')
        print('DONE')
        # for session_id, group in tqdm(grouped):
        #     sess_item_id = LabelEncoder().fit_transform(group.item_id)
        #     group = group.reset_index(drop=True)
        #     group['sess_item_id'] = sess_item_id
        #     node_features = group.loc[group.session_id==session_id,['sess_item_id','item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values
        #
        #     node_features = torch.LongTensor(node_features).unsqueeze(1)
        #     target_nodes = group.sess_item_id.values[1:]
        #     source_nodes = group.sess_item_id.values[:-1]
        #
        #     edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        #     x = node_features
        #
        #     y = torch.FloatTensor([group.label.values[0]])
        #
        #     data = Data(x=x, edge_index=edge_index, y=y)
        #     data_list.append(data)
        #
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])