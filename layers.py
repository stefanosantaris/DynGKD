import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
class StructuralSelfAttention(Module):
    def __init__(self, input_dimension, output_dimension, n_heads, dropout, alpha):
        super(StructuralSelfAttention,self).__init__()
        self.GAT = GAT(input_dimension, output_dimension, output_dimension, dropout, alpha, n_heads)

    def forward(self,feat, adj):
        return self.GAT(feat, adj)



class GAT(Module):
    def __init__(self, in_features, n_hidden, n_out, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GATLayer(in_features, n_hidden, dropout, alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(n_hidden * n_heads, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        # in_features denoted by F
        self.in_features = in_features
        # out_features denoted by F'
        self.out_features = out_features

        self.alpha = alpha
        self.concat = concat
        #\mathbf{W} \in \mathbb{R}^{F \times F'}
        self.W  = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #\mathbf{h} \in \mathbb{R}^{N \times F}
        #\mathbf{Wh} \in \mathbb{R}^{N \times F'}
        Wh = torch.mm(h, self.W)

        edges = adj._indices()
        attention_input = torch.cat((Wh[edges[0,:],:], Wh[edges[1,:],:]), dim=1)
        e = self.leakyrelu(torch.mm(attention_input, self.a)).squeeze()

        # coefficients = torch.zeros(size=(N,N))

        coefficients = torch.sparse_coo_tensor(edges, e, adj.shape, requires_grad=False)
        attention = torch.sparse.softmax(coefficients, dim=1).to_dense()

        # a_input = self.__prepare_attention_input(Wh)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #
        #
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __prepare_attention_input(self, Wh):
        #Number of nodes
        N = Wh.size()[0]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N,1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)



