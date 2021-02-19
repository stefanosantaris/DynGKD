import argparse
from torch.nn.modules import loss
import torch.optim as optim
from numpy import dtype
from graph_dataset import GraphDataset
import scipy
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import defaultdict
from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data, normalize_graph_gcn, preprocess_features
from models import DynGKD
from torch.utils.data import DataLoader

def preprocess(args):
    window = args.window
    predict_snapshot_id = args.predict_snapshot
    start_snapshot_id = max(0, predict_snapshot_id - 1 - window)

    graphs, adjs = load_graphs(args.dataset_file, start_snapshot_id, predict_snapshot_id)
    assert predict_snapshot_id - start_snapshot_id == len(adjs) 
    
    # if args.featureless:
    feats = [scipy.sparse.identity(adjs[-1].shape[0], dtype=np.float32).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[-1].shape[0]]
    num_features = feats[0].shape[1]

    context_pairs_train = get_context_pairs(args.dataset_file, start_snapshot_id, predict_snapshot_id)

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(adjs)

    new_G = nx.MultiGraph()
    new_G.add_nodes_from(graphs[-1].nodes(data=True))

    for e in graphs[-2].edges():
        new_G.add_edge(e[0],e[1])

    graphs[-1] = new_G
    adjs[-1] = nx.adjacency_matrix(new_G)


    print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

    # Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
    adj_train = [normalize_graph_gcn(adj) for adj in adjs]
    # [*map(lambda adj: normalize_graph_gcn(adj), adjs)]

    feats_train = [preprocess_features(feat) for feat in feats]
    #  [*map(lambda feat: preprocess_features(feat)[1], feats)]

    print("Build the graph dataset")
    graph_dataset = GraphDataset(graphs, feats_train, adj_train, context_pairs_train, negative_sample=1, window=2)

    dataloader = DataLoader(graph_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    structural_layer_config = [int(t) for t in args.structural_layer_config.split(',')]
    structural_head_config = [int(t) for t in args.structural_head_config.split(',')]

    temporal_layer_config = [int(t) for t in args.temporal_layer_config.split(',')]
    temporal_head_config = [int(t) for t in args.temporal_head_config.split(',')]

    model = DynGKD(num_features, structural_layer_config, structural_head_config, temporal_layer_config, temporal_head_config, len(adjs))
    print("Number of parameters " + str(count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        epoch_loss = 0
        for result in dataloader:
            optimizer.zero_grad()
            scores=[]
            final_output_embeddings = model(result['feats'], result['adjs'])
            for t in range(len(adjs)):
                output_embeds_t = final_output_embeddings.permute(1,0,2)[t]
                inputs_1 = torch.index_select(output_embeds_t, 0, result['node_1'][t])
                inputs_2 = torch.index_select(output_embeds_t, 0, result['node_2'][t])
                scores.append(torch.sum(torch.mul(inputs_1, inputs_2), dim=0, keepdim=True))
            pos_scores = torch.stack(scores)
            criterion = nn.MSELoss()
            R_loss = criterion(torch.ones_like(pos_scores), pos_scores)
            loss_train = torch.sqrt(R_loss)
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train
        print("Epoch {} loss {} ".format(epoch, epoch_loss))

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
    parser.add_argument('--dataset_file',type=str, default='Enron_new', help='Dataset path')
    parser.add_argument('--featureless', type=bool, default=True, help='Node features')
    parser.add_argument('--predict_snapshot', type=int, default=3, help='Graph snapshot that we try to predict')
    parser.add_argument('--window', type=int, default=5, help='Window of previous graph snapshots that we want to include in training')
    parser.add_argument('--negative_sample', type=int, default=1, help="Negative Sampling")
    parser.add_argument('--structural_layer_config', type=str, default='256', help="Dimension of each GAT layer. Seprated by comma")
    parser.add_argument('--structural_head_config', type=str, default='16', help='Number of heads applied in each GAT layer')
    parser.add_argument('--temporal_layer_config', type=str, default='128', help="Dimnesion of each temporal attention layer. Separated by comma")
    parser.add_argument('--temporal_head_config', type=str, default='16', help='Number of heads appliad in each attention layer')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    args = parser.parse_args()
    preprocess(args)