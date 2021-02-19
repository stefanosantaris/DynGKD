from torch.nn.modules.module import Module
from layers import StructuralSelfAttention
class DynGKD(Module):
    def __init__(self, n_feats, structural_layer_config, structural_head_config):
        super(DynGKD,self).__init__()
        self.structural_layer_config = structural_layer_config
        self.num_structural_layers = len(structural_layer_config)
        self.structural_head_config = structural_head_config
        self.n_feats = n_feats

        assert self.num_structural_layers == len(structural_head_config)
        self.structural_layers = []
        for i in range(self.num_structural_layers):
            input_dimension = self.n_feats  if i == 0 else self.structural_layer_config[i - 1]
            output_dimension = self.structural_layer_config[i]
            structural_layer = StructuralSelfAttention(input_dimension, output_dimension, self.structural_head_config[i], dropout=0., alpha=0.2)
            self.structural_layers.append(structural_layer)
            self.add_module('structural_layer_{}'.format(i), structural_layer)



    def forward(self, feats, adjs):
        structural_outputs = []
        for i,feat in enumerate(feats):
            out = feat
            for layer in self.structural_layers:
                out = layer(out,adjs[i])
            structural_outputs.append(out)
        return structural_outputs

