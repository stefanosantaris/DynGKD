import networkx as nx
from collections import defaultdict

from networkx.generators.joint_degree_seq import _directed_neighbor_switch_rev
from utils.random_walk import Graph_RandomWalk
import random

def run_random_walks_n2v(graph, num_walks=10, walk_len=40):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    nx_G = nx.Graph()
    nx_G.add_weighted_edges_from([(edge[0],edge[1],edge[2]['weight']) for edge in graph.edges(data=True)])
    # adj = nx.adjacency_matrix(graph)
    # for e in graph.edges():
    #     nx_G.add_edge(e[0], e[1])

    # for edge in graph.edges():
    #     nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    print('Start Random Walks')
    walks = build_walk_corpus(nx_G, num_walks, walk_len)


    # G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    # print("Preprocess transition probs")
    # G.preprocess_transition_probs()
    # print("Simulate Walks")
    # walks = G.simulate_walks(num_walks, walk_len)
    # print("Finished Random Walk")
    WINDOW_SIZE = 10
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            flag = False
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
                    flag=True
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs


def random_walk(graph, start_node, walk_length, rand):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        if len(graph[cur]) > 0:
            neighbors = [n for n in graph.neighbors(cur)]
            rand.shuffle(neighbors)
            walk.append(neighbors[0])
        else:
            print("No neighbors")
            break
    return walk


def build_walk_corpus(graph, num_walks, walk_length):
    nodes = list(graph.nodes())
    rand = random.Random(0)
    walks = []
    for cnt in range(num_walks):
        rand.shuffle(nodes)
        for i, node in enumerate(nodes):
            walks.append(random_walk(graph, node, walk_length,rand))
    return walks