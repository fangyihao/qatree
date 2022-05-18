'''
Created on May 9, 2022

@author: Yihao Fang
'''
from neural_tree.models import BasicNetwork
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from transformers import RobertaModel, MobileBertModel
from torch import nn
import torch
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
        encoder.config.pre_seq_len = 8
        
        #encoder = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        super(PrefixNeuralTreeNetwork, self).__init__(None, None, task, conv_block, None, num_layers,
                                                None, GAT_heads, GAT_concats, dropout, encoder = encoder)
        self.encoder = encoder
        self.num_choices = 5
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, data):
        x, token_type_ids, edge_index, batch, nc_mask = data.x, data.node_token_type_ids, data.edge_index, data.batch, data.nc_mask
        
        #print("x:", list(x.cpu().numpy()))
        #print("edge_index:", list(edge_index.cpu().numpy()))
        #print("data.leaf_mask:", list(data.leaf_mask.cpu().numpy()))
        #print("batch*self.num_choices + nc_mask:", list((batch*self.num_choices + nc_mask).cpu().numpy()))
        
        if data.num_node_features == 0:
            raise RuntimeError('No node feature')
        
        x_len = x.size(0)
        
        x_index = torch.arange(x.size(0)).long()
        x_index = torch.cat([x_index[data.leaf_mask], x_index[torch.logical_not(data.leaf_mask)]])
        x_index = torch.argsort (x_index)
        
        x = self.encoder.embeddings(
            input_ids=x[data.leaf_mask].long(),
            token_type_ids=token_type_ids[data.leaf_mask].long()
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.cat([x, torch.zeros(x_len-x.size(0), *x.size()[1:]).to(device)])[x_index]
        
        
        #print("x:", [list(x) for x in list(x.cpu().numpy())])

        if not self.need_postmp:  # pre-iteration dropout for citation networks (might not be necessary in some case)
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index=edge_index)
            #print("x:", x.cpu().detach().numpy())
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
