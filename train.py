import argparse

from numpy import dtype
from graph_dataset import GraphDataset
import scipy
import torch
import networkx as nx
import numpy as np
from collections import defaultdict
from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data, normalize_graph_gcn, preprocess_features
from models import DynGKD
from torch.utils.data import DataLoader

def preprocess(args):
    num_time_steps = args.predict_snapshot
    graphs, adjs = load_graphs(args.dataset_file)
    assert num_time_steps < len(adjs) + 1
    
    if args.featureless:
        feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0], dtype=np.float32).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
    num_features = feats[0].shape[1]

    context_pairs_train = get_context_pairs(args.dataset_file, graphs, num_time_steps)

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(adjs, num_time_steps, args.dataset_file)

    new_G = nx.MultiGraph()
    new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

    for e in graphs[num_time_steps - 2].edges():
        new_G.add_edge(e[0],e[1])

    graphs[num_time_steps - 1] = new_G
    adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)


    print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

    # Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
    adj_train = [normalize_graph_gcn(adj) for adj in adjs]
    # [*map(lambda adj: normalize_graph_gcn(adj), adjs)]

    feats_train = [preprocess_features(feat) for feat in feats]
    #  [*map(lambda feat: preprocess_features(feat)[1], feats)]

    print("Build the graph dataset")
    graph_dataset = GraphDataset(graphs, feats_train, adj_train, num_time_steps, context_pairs_train, negative_sample=1, window=2)

    dataloader = DataLoader(graph_dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_batch)

    structural_layer_config = [int(t) for t in args.structural_layer_config.split(',')]
    structural_head_config = [int(t) for t in args.structural_head_config.split(',')]

    model = DynGKD(num_features, structural_layer_config, structural_head_config)
    print("Number of parameters " + str(count_parameters(model)))
    for _ in range(3):
        for result in dataloader:
            test = model(result['feats'], result['adjs'])
            # print(test)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_batch(batch):
    node_1_dict = defaultdict(lambda: [])
    node_2_dict = defaultdict(lambda: [])
    for d in batch:
        node_1 = d['node_1']
        node_2 = d['node_2']
        for t in range(len(node_1)):
            node_1_dict[t].extend(node_1[t])
            node_2_dict[t].extend(node_2[t])
    node_1_batch = [torch.tensor(node_1_dict[k]) for k in node_1_dict.keys()]
    node_2_batch = [torch.tensor(node_2_dict[k]) for k in node_2_dict.keys()]

    feats = [torch.sparse.FloatTensor(torch.from_numpy(t[0].transpose().astype(np.int64)), torch.from_numpy(t[1]), torch.Size(t[2])) for t in batch[0]['feats']]
    adjs = [torch.sparse.FloatTensor(torch.from_numpy(t[0].transpose().astype(np.int64)), torch.from_numpy(t[1]), torch.Size(t[2])) for t in batch[0]['adjs']]
        
    return {'node_1':node_1_batch, 'node_2':node_2_batch, 'feats':feats, 'adjs': adjs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file',type=str, default='ml-10m_new', help='Dataset path')
    parser.add_argument('--featureless', type=bool, default=True, help='Node features')
    parser.add_argument('--predict_snapshot', type=int, default=5, help='Graph snapshot that we try to predict')
    parser.add_argument('--negative_sample', type=int, default=1, help="Negative Sampling")
    parser.add_argument('--structural_layer_config', type=str, default='128', help="Dimension of each GAT layer. Seprated by comma")
    parser.add_argument('--structural_head_config', type=str, default='2', help='Number of heads applied in each GAT layer')



    args = parser.parse_args()
    preprocess(args)