import torch
import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GraphDataset(Dataset):
    def __init__(self, graphs, max_num_nodes, negative_sample = 1, window=0):
        self.graphs = graphs[:-1]
        self.negative_sample = negative_sample
        self.window = window
        self.train_nodes = np.arange(max_num_nodes)
        self.train_nodes = np.random.permutation(self.train_nodes)


    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        n = self.train_nodes[idx]

        node_1_all = []
        node_2_all = []
        weight_all = []
        for t in range(0, len(self.graphs)):
            node_1 = []
            node_2 = []
            weights = []
            if n in set(self.graphs[t].nodes()):
                if len(list(self.graphs[t].neighbors(n))) > self.negative_sample:
                    node_1_sample = [n for _ in range(self.negative_sample)]
                    node_2_sample = [d for d in np.random.choice(list(self.graphs[t].neighbors(n)), self.negative_sample, replace=False)]
                    weight_sample = [self.graphs[t][src][dst]['weight'] for src, dst in list(zip(*[node_1_sample, node_2_sample]))]
                elif len(list(self.graphs[t].neighbors(n))) > 0:
                    node_1_sample = [n for _ in list(self.graphs[t].neighbors(n))]
                    node_2_sample = [d for d in list(self.graphs[t].neighbors(n))]
                    weight_sample = [self.graphs[t][src][dst]['weight'] for src, dst in list(zip(*[node_1_sample, node_2_sample]))]

                node_1.extend(node_1_sample)
                node_2.extend(node_2_sample)
                weights.extend(weight_sample)
            
            assert len(node_1) == len(node_2)
            assert len(node_1) == len(weights)
            assert len(node_1) <= self.negative_sample

            node_1_all.append(node_1)
            node_2_all.append(node_2)
            weight_all.append(weights)
        result = {'node_1':node_1_all, 'node_2':node_2_all, 'weights': weight_all}
        # print(result)

        return result





        