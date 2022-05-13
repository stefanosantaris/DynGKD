import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
import random
from utils.utilities import run_random_walks_n2v


def load_graphs(dataset_str, start_id, stop_id, one_file = False, edge_list=False):
    # graphs = []
    # for i in range(start_id, stop_id):
    #     graphs.append(nx.read_gexf)
    if one_file:
        graphs = np.load("data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True, encoding='latin1')['graph']
        graphs = [nx_graph(g) for g in graphs][start_id:stop_id]
    elif edge_list:
        graphs = [nx.read_weighted_edgelist(f'data/{dataset_str}/edges{i}.csv', delimiter=',', nodetype=int,encoding='utf-8') for i in np.arange(start_id, stop_id)]
    else:
        graphs = [nx.read_gpickle("data/{}/graph_{}.npz".format(dataset_str, i)) for i in range(start_id, stop_id)]
        graphs = [nx_graph(g) for g in graphs]
    # graphs = np.load("data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True)['graph']
    print("Loaded {} graphs ".format(len(graphs)))
    adj_matrices = [*map(lambda x: nx.adjacency_matrix(x), graphs)]
    return graphs, adj_matrices

def nx_graph(graph, one_file=True):
        nx_G = nx.Graph()
        if one_file:
            nx_G.add_nodes_from(list(graph.node))
            for src in graph.edge:
                for dst in graph.edge[src]:
                    nx_G.add_edge(src, dst, weight=1)
            # nx_G.add_weighted_edges_from([(edge[0],edge[1],edge[2]['weight']) for edge in graph.edge])
        else:
            nx_G.add_nodes_from(list(graph.nodes()))
            nx_G.add_weighted_edges_from([(edge[0],edge[1],edge[2]['weight']) for edge in graph.edges(data=True)])
        return nx_G


def get_context_pairs(dataset, start_id, stop_id):
    context_pair_train = [pkl.load(open("data/{}/pairs_{}.pkl".format(dataset, i), 'rb')) for i in range(start_id, stop_id)]
    return context_pair_train


def get_evaluation_data(graphs, num_nodes, train_perc = 0.6, val_perc = 0.2, ns = 1):
    """ Load train/val/test examples to evaluate link prediction performance"""

    next_graph = graphs[-1]
    graph = graphs[-2]

    print("Generating and saving eval data ....")
    train_graph_edges = set(graph.edges())
    next_graph_edges = set(next_graph.edges())

    all_edges = train_graph_edges.union(next_graph_edges)

    new_edges_list = list(next_graph_edges - train_graph_edges)
    random.shuffle(new_edges_list)

    train_edges_num = int(len(new_edges_list) * train_perc)
    train_edges = new_edges_list[:train_edges_num]
    train_edges_false = generate_ns(train_edges, all_edges, num_nodes, ns)


    val_edges_num = int(len(new_edges_list) * (train_perc + val_perc))
    val_edges = new_edges_list[train_edges_num+1:val_edges_num]
    val_edges_false = generate_ns(val_edges, all_edges, num_nodes, ns)

    test_edges = new_edges_list[val_edges_num + 1:]
    test_edges_false = generate_ns(test_edges, all_edges, num_nodes, ns)

    # train_edges_ns = generate_ns(list(train_graph_edges), all_edges, self.num_nodes, ns)
    train_edges = [next_graph[src][dst]['weight'] for (src,dst) in train_edges]
    val_edges = [next_graph[src][dst]['weight'] for (src,dst) in val_edges]
    test_edges = [next_graph[src][dst]['weight'] for (src,dst) in test_edges]


    
    # train_edges_tuples = [(edge[0],edge[1],edge[2]['weight']) for edge in graph.edges(data=True)]

    # train_adj = generate_adjacency_matrix(train_edges_tuples, self.num_nodes)

    # train_adj_norm = normalize_adj(train_adj)

    # train_edges_with_ns = train_edges_tuples
    # train_edges_with_ns.extend(train_edges_ns)

    # val_edges_with_ns = [(src, dst, next_graph[src][dst]['weight']) for (src,dst) in val_graph_edges]
    # val_edges_with_ns.extend(val_graph_edges_ns)

    # test_edges_with_ns = [(src, dst, next_graph[src][dst]['weight']) for (src, dst) in test_graph_edges]
    # test_edges_with_ns.extend(test_graph_edges_ns)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false
    # train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    #         create_data_splits(graphs[-2], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)


    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def generate_ns(positive_edges, edges_no_weights, num_nodes, ns=1):
    ns_dict = dict()
    while len(ns_dict) < len(positive_edges) * ns:
        src = random.randint(0, num_nodes-1)
        dst = random.randint(0, num_nodes-1)
        if src!=dst \
                and (src,dst) not in ns_dict \
                and (src,dst) not in edges_no_weights:
                ns_dict[(src,dst)] = 0
    return [(src,dst,0) for (src,dst),value in ns_dict.items()]

def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false

def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)