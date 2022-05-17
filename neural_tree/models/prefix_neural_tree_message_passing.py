'''
Created on May 9, 2022

@author: Yihao Fang
'''
from neural_tree.models import BasicNetwork
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from transformers import RobertaModel, MobileBertModel
from torch import nn
class PrefixNeuralTreeNetwork(BasicNetwork):
    def __init__(self, input_dim, output_dim, task='node', conv_block='GCN', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25):
        """
        NeuralTreeNetwork is the child class of BasicNetwork, which implements basic message passing on graphs.
        The network parameters and loss functions are the same as the parent class. The difference is that this class
         has an additional pooling layer at the end to aggregate final hidden states of the leaf nodes (for node
         classification).
        """
        encoder = RobertaModel.from_pretrained("roberta-base")
        
        for param in encoder.parameters():
            param.requires_grad = False
            
        encoder.config.prefix_projection = False
        encoder.config.pre_seq_len = 4
        
        #encoder = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        super(PrefixNeuralTreeNetwork, self).__init__(None, None, task, conv_block, None, num_layers,
                                                None, GAT_heads, GAT_concats, dropout, encoder = encoder)
        self.encoder = encoder
        self.num_choices = 5
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, data):
        x, token_type_ids, edge_index, batch, nc_mask = data.x, data.node_token_type_ids, data.edge_index, data.batch, data.nc_mask
        
        
        if data.num_node_features == 0:
            raise RuntimeError('No node feature')
        
        x = self.encoder.embeddings(
            input_ids=x.long(),
            token_type_ids=token_type_ids.long()
        )

        if not self.need_postmp:  # pre-iteration dropout for citation networks (might not be necessary in some case)
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index=edge_index)
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                if self.conv_block == 'GIN':
                    x = self.batch_norms[i](x)
                x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.encoder.pooler(x)

        x = pyg_nn.global_mean_pool(x[data.leaf_mask, :], batch[data.leaf_mask]*self.num_choices + nc_mask[data.leaf_mask])
        
        
        x = self.classifier(x)
        x = x.view(-1, self.num_choices)
        
        #print("x.size():", x.size())

        if self.need_postmp:
            x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            return tuple(self.post_mp[i](x) for i in range(len(self.post_mp)))
        else:
            return x
