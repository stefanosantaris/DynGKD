import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GraphDataset(Dataset):
    def __init__(self, graphs, features, adjs, context_pairs, negative_sample = 1, window=0):
        self.graphs = graphs
        self.features = features
        self.adjs = adjs
        self.context_pairs = context_pairs
        self.negative_sample = negative_sample
        self.window = window
        self.train_nodes = list(self.graphs[-1].nodes())
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.degs = self.construct_degs()
    

    def construct_degs(self):
        degs = []
        for i in range(0, len(self.graphs)):
            G = self.graphs[i]
            deg = np.zeros((G.number_of_nodes(),))
            for node in G.nodes():
                neighbors = np.array(list(G.neighbors(node)))
                deg[node] = len(neighbors)
            degs.append(deg)
        return degs


    def __len__(self):
        return self.graphs[-1].number_of_nodes() - 1

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        n = self.train_nodes[idx]

        node_1_all = []
        node_2_all = []
        features_all = []
        adjs_all = []
        # batch_nodes = self.train_nodes[idx:idx+self.batch_size]
        for t in range(0, len(self.graphs)):
            node_1 = []
            node_2 = []
            if n in self.context_pairs[t]:
                if len(self.context_pairs[t][n]) > self.negative_sample:
                    node_1.extend([n] * self.negative_sample)
                    node_2.extend(np.random.choice(self.context_pairs[t][n], self.negative_sample, replace=False))
                elif len(self.context_pairs[t][n]) > 0:
                    node_1.extend([n] * len(self.context_pairs[t][n]))
                    node_2.extend(self.context_pairs[t][n])
            
            assert len(node_1) == len(node_2)
            assert len(node_1) <= self.negative_sample

            node_1_all.append(node_1)
            node_2_all.append(node_2)
            features_all.append(self.features[t])
            adjs_all.append(self.adjs[t])
        result = {'node_1':node_1_all, 'node_2':node_2_all, 'feats':features_all, 'adjs':adjs_all}
        # print(result)

        return result





        