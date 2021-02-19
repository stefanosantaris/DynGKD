import torch
from torch.nn.modules.module import Module
from layers import StructuralSelfAttention, TemporalAttention
class DynGKD(Module):
    def __init__(self, n_feats, structural_layer_config, structural_head_config, temporal_layer_config, temporal_head_config, num_snapshots):
        super(DynGKD,self).__init__()
        self.structural_layer_config = structural_layer_config
        self.num_structural_layers = len(structural_layer_config)
        self.structural_head_config = structural_head_config
        
        self.temporal_layer_config  = temporal_layer_config
        self.num_temporal_layers = len(self.temporal_layer_config)
        self.temporal_head_config = temporal_head_config
        self.n_feats = n_feats

        assert self.num_structural_layers == len(structural_head_config)
        assert self.num_temporal_layers == len(temporal_layer_config)


        self.structural_layers = []
        for i in range(self.num_structural_layers):
            input_dimension = self.n_feats  if i == 0 else self.structural_layer_config[i - 1]
            output_dimension = self.structural_layer_config[i]
            structural_layer = StructuralSelfAttention(input_dimension, output_dimension, self.structural_head_config[i], dropout=0., alpha=0.2)
            self.structural_layers.append(structural_layer)
            self.add_module('structural_layer_{}'.format(i), structural_layer)


        self.temporal_layers = []
        for i in range(self.num_temporal_layers):
            input_dimension = self.structural_layer_config[-1] if i ==0 else self.temporal_layer_config[i - 1]
            output_dimension = self.temporal_layer_config[i]
            temporal_layer = TemporalAttention(input_dim=input_dimension, output_dim=output_dimension, n_heads=self.temporal_head_config[i], num_snapshots=num_snapshots, dropout=0., alpha=0.2)
            self.temporal_layers.append(temporal_layer)
            self.add_module('temporal_layer_{}'.format(i), temporal_layer)



    def forward(self, feats, adjs):
        # 1. Compute structural outputs of each graph snapshot
        structural_outputs = []
        for i,feat in enumerate(feats):
            out = feat
            for layer in self.structural_layers:
                out = layer(out,adjs[i])
            structural_outputs.append(out)

        # 2. Pack structural embeddings across snapshots
        attention_outputs = []
        for t in range(0, len(feats)):
            zero_padding = torch.zeros(structural_outputs[-1].shape[0] - structural_outputs[t].shape[0], self.structural_layer_config[-1])
            attention_outputs.append(torch.cat((structural_outputs[t], zero_padding)))
        structural_embeddings = torch.stack(attention_outputs).permute(1,0,2)

        # 3. Compute the temporal embeddings
        temporal_embeddings = structural_embeddings
        for i, layer in enumerate(self.temporal_layers):
            temporal_embeddings = layer(temporal_embeddings)
        
        return temporal_embeddings

