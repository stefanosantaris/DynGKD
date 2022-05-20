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
import time



def preprocess(args):
    window = args.window
    predict_snapshot_id = args.predict_snapshot
    start_snapshot_id = max(0, predict_snapshot_id - 1 - window)

    graphs, adjs = load_graphs(args.dataset_file, start_snapshot_id, predict_snapshot_id, one_file=False, edge_list=True)
    assert predict_snapshot_id - start_snapshot_id == len(adjs) 
    max_num_nodes = max([graph.number_of_nodes() for graph in graphs])
    features_basis = scipy.sparse.identity(max_num_nodes, dtype=np.float32).tocsr()
    
    if args.featureless:
    # global feats
        feats = [preprocess_features(features_basis[range(0, x.shape[0]), :]) for x in adjs]

    # context_pairs_train = get_context_pairs(args.dataset_file, start_snapshot_id, predict_snapshot_id)

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(graphs, max_num_nodes)

    new_G = nx.MultiGraph()
    new_G.add_nodes_from(graphs[-1].nodes(data=True))

    for _ in graphs[-2].edges(data=True):
        new_G.add_weighted_edges_from([(edge[0],edge[1],edge[2]['weight']) for edge in graphs[-2].edges(data=True)])

    graphs[-1] = new_G
    adjs[-1] = nx.adjacency_matrix(new_G)


    print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

    # Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
    # global adj_train
    adj_train = [normalize_graph_gcn(adj) for adj in adjs]
    # [*map(lambda adj: normalize_graph_gcn(adj), adjs)]

    feats_train = [preprocess_features(feat) for feat in feats]
    #  [*map(lambda feat: preprocess_features(feat)[1], feats)]

    print("Build the graph dataset")
    graph_dataset = GraphDataset(graphs, max_num_nodes, negative_sample=args.negative_sample, window=2)

    dataloader = DataLoader(graph_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    structural_layer_config = [int(t) for t in args.structural_layer_config.split(',')]
    structural_head_config = [int(t) for t in args.structural_head_config.split(',')]

    temporal_layer_config = [int(t) for t in args.temporal_layer_config.split(',')]
    temporal_head_config = [int(t) for t in args.temporal_head_config.split(',')]

    model = DynGKD(max_num_nodes, structural_layer_config, structural_head_config, temporal_layer_config, temporal_head_config, len(adjs[:-1]))
    # teacher_model = torch.load(f'{path}')
    # print("Predict snap "+ str(args.predict_snapshot))
    print("Number of parameters " + str(count_parameters(model)))
    
    if not args.teacher:
        teacher_model = torch.load(f'/models/{args.dataset_file}_teacher.pt')
    # print("Number of Nodes " + str(graphs[-1].number_of_nodes()))

    inference_times = []
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        epoch_loss = 0
        for result in dataloader:
            optimizer.zero_grad()
            scores=[]
            weights = []
            teacher_scores = []
            teacher_weights = []
            start_ts = time.time()
            final_output_embeddings = model(result['feats'], result['adjs'])
            
            if not args.teacher:
                teacher_output_embeddings = teacher_model(result['feats'], result['adjs'])
            
            # final_teacher_embeddings = teacher_model(result['feats'],result['adjs'])
            end_ts = time.time()
            inference_times.append(end_ts - start_ts)
            print('inference time ' + str((end_ts - start_ts)))

            for t in range(len(adjs[:-1])):
                output_embeds_t = final_output_embeddings.permute(1,0,2)[t]
                inputs_1 = torch.index_select(output_embeds_t, 0, result['node_1'][t])
                inputs_2 = torch.index_select(output_embeds_t, 0, result['node_2'][t])
                scores.append(torch.sum(torch.mul(inputs_1, inputs_2), dim=1, keepdim=True))
                weights.append(result['weights'][t])
                
                if not args.teacher:
                    output_embeds_t = teacher_output_embeddings.permute(1,0,2)[t]
                    inputs_1 = torch.index_select(output_embeds_t, 0, result['node_1'][t])
                    inputs_2 = torch.index_select(output_embeds_t, 0, result['node_2'][t])
                    teacher_scores.append(torch.sum(torch.mul(inputs_1, inputs_2), dim=1, keepdim=True))
                    teacher_weights.append(result['weights'][t])
                    
            criterion = nn.MSELoss()
            student_loss = torch.mean(torch.stack([criterion(torch.reshape(weights[t], (scores[t].shape[0],1)), scores[t]) for t in range(len(adjs))]), dtype=torch.float)
            if not args.teacher and args.loss_type == 0:
                teacher_loss = torch.mean(torch.stack([criterion(torch.reshape(teacher_weights[t], (teacher_scores[t].shape[0],1)), teacher_scores[t]) for t in range(len(adjs))]), dtype=torch.float)
                
                final_loss = args.gamma * student_loss + (1-args.gamma) * teacher_loss 
            else:
                final_loss = args.gamma * student_loss  
                     
            # R_loss = criterion(torch.ones_like(pos_scores), pos_scores)
            loss_train = torch.sqrt(final_loss)
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train
        print("Epoch {}".format(epoch))
    avg_time = np.mean(inference_times)
    print(f"Inference time Average {avg_time}")
    
    # If we have trained the teacher model, then we save it with the teacher extension.
    if args.teacher:
        torch.save(model, f'/models/{args.dataset_file}_teacher.pt')
    else:
        torch.save(model, f"/models/{args.dataset_file}_student.pt")
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_batch(batch):
    node_1_dict = defaultdict(lambda: [])
    node_2_dict = defaultdict(lambda: [])
    weights_dict = defaultdict(lambda: [])
    for d in batch:
        node_1 = d['node_1']
        node_2 = d['node_2']
        weight = d['weights']
        for t in range(len(node_1)):
            node_1_dict[t].extend(node_1[t])
            node_2_dict[t].extend(node_2[t])
            weights_dict[t].extend(weight[t])
    node_1_batch = [torch.tensor(node_1_dict[k]) for k in node_1_dict.keys()]
    node_2_batch = [torch.tensor(node_2_dict[k]) for k in node_2_dict.keys()]
    weight_batch = [torch.tensor(weights_dict[k], dtype=torch.float) for k in weights_dict.keys()]
    num_feats = len(batch[0]['node_1'])

    batch_feats = [torch.sparse.FloatTensor(torch.from_numpy(t[0].transpose().astype(np.int64)), torch.from_numpy(t[1]), torch.Size(t[2])) for t in feats[-num_feats-1:-1]]
    batch_adjs = [torch.sparse.FloatTensor(torch.from_numpy(t[0].transpose().astype(np.int64)), torch.from_numpy(t[1]), torch.Size(t[2])) for t in adj_train[-num_feats-1:-1]]
        
    return {'node_1':node_1_batch, 'node_2':node_2_batch, 'weights':weight_batch, 'feats':batch_feats, 'adjs': batch_adjs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file',type=str, default='livestream-4k', help='Dataset path')
    parser.add_argument('--teacher', type=bool, default=True, help="Teacher model")
    parser.add_argument('--loss_type', type=int, default=0, help="Loss function")
    parser.add_argument('--featureless', type=bool, default=True, help='Node features')
    parser.add_argument('--predict_snapshot', type=int, default=5, help='Graph snapshot that we try to predict')
    parser.add_argument('--window', type=int, default=7, help='Window of previous graph snapshots that we want to include in training')
    parser.add_argument('--negative_sample', type=int, default=5, help="Negative Sampling")
    parser.add_argument('--structural_layer_config', type=str, default='256', help="Dimension of each GAT layer. Seprated by comma")
    parser.add_argument('--structural_head_config', type=str, default='12', help='Number of heads applied in each GAT layer')
    parser.add_argument('--temporal_layer_config', type=str, default='256', help="Dimnesion of each temporal attention layer. Separated by comma")
    parser.add_argument('--temporal_head_config', type=str, default='12', help='Number of heads appliad in each attention layer')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma variable")
    args = parser.parse_args()
    preprocess(args)