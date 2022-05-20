import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
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



class TemporalAttention(Module):
    def __init__(self, input_dim, output_dim, n_heads, num_snapshots, dropout, alpha):
        super(TemporalAttention,self).__init__()
        self.num_snapshots = num_snapshots
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.position_embeddings = nn.Embedding(num_snapshots, input_dim)

        self.q_embedding_weights = Parameter(torch.FloatTensor(input_dim, input_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.q_embedding_weights)
        self.k_embedding_weights = Parameter(torch.FloatTensor(input_dim, input_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.k_embedding_weights)
        self.v_embedding_weights = Parameter(torch.FloatTensor(input_dim, input_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.v_embedding_weights)
        
    def forward(self, input):
        position_inputs = torch.unsqueeze(torch.arange(0, self.num_snapshots, dtype=torch.int64),0).repeat((input.shape[0], 1))
        temporal_inputs = input + self.position_embeddings(position_inputs)

        q = torch.tensordot(temporal_inputs, self.q_embedding_weights, dims=[[2],[0]])
        k = torch.tensordot(temporal_inputs, self.k_embedding_weights, dims=[[2],[0]])
        v = torch.tensordot(temporal_inputs, self.v_embedding_weights, dims=[[2],[0]])

        q_ = torch.cat(torch.chunk(q, self.n_heads, dim=2), dim=0)
        k_ = torch.cat(torch.chunk(k, self.n_heads, dim=2), dim=0)
        v_ = torch.cat(torch.chunk(v, self.n_heads, dim=2), dim=0)

        outputs = torch.matmul(q_, k_.permute(0,2,1))
        outputs = outputs / (self.num_snapshots ** 0.5)
        
        diag_val = torch.ones_like(outputs[0,:,:])
        tril = torch.tril(diag_val, diagonal=0)
        masks = torch.unsqueeze(tril,0).repeat((outputs.shape[0], 1, 1))
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)

        outputs = torch.where(masks > 0, padding, outputs)
        outputs = F.softmax(outputs, dim=0)

        outputs = torch.matmul(outputs, v_)

        split_outputs = torch.chunk(outputs, self.n_heads, dim=0)
        outputs = torch.cat(split_outputs, dim=-1)

        test = torch.reshape(outputs, (-1, self.num_snapshots, self.input_dim))

        return outputs