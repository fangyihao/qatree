'''
Created on Apr. 25, 2022

@author: Yihao Fang
'''
from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, Size

from transformers.models.albert.modeling_albert import AlbertLayer
from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer
from transformers import AutoConfig
from torch import nn
from transformers import RobertaModel
import torch
from model.prefix_encoder import PrefixEncoder
from typing import Optional
from transformers.models.deberta_v2.modeling_deberta_v2 import build_relative_position
from mpu.transformer import ParallelTransformerLayer
import numpy as np
import torch.utils.checkpoint
import itertools
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import networkx as nx
import random

class SAGEFormer2D(MessagePassing):
    
    def __init__(self, config, normalize: bool = False, non_prefix_requires_grad:bool = False, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        
        super().__init__(**kwargs)
        
        self.config = config
        
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if config.name_or_path.startswith("bert") or "SapBERT" in config.name_or_path:
            cls = BertLayer
        #elif config.name_or_path.startswith("roberta"):
        elif "roberta" in config.name_or_path:
            cls = RobertaLayer
        elif config.name_or_path.startswith("albert"):
            cls = AlbertLayer
        elif "deberta" in config.name_or_path:
            cls = DebertaV2Layer
        elif config.name_or_path.startswith("glm"):    
            cls = ParallelTransformerLayer 
        else:
            raise RuntimeError("Not implemented")
        
        self.layer_module = cls(config)
        
        if config.prefix_tuning == True:
            for param in self.layer_module.parameters():
                param.requires_grad = non_prefix_requires_grad
        
        self.normalize = normalize
        
        
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads
        
        if config.prefix_tuning == True:
            self.pre_seq_len = self.config.pre_seq_len
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_tokens.to(device)
            
            self.prefix_encoder = PrefixEncoder(self.config)
            self.prefix_encoder.to(device)
        
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        #self.dropout.to(device)
        
    def create_prefix(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        
        return past_key_values

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q, hidden_states.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions
            )
        return relative_pos
    
    def random_walk(self, edge_index: Adj, edge_weight: Tensor, context_index, sample_rate):
        
        device = edge_index.device
        
        data = Data(edge_index = edge_index, edge_weight = edge_weight)
        G = pyg_utils.to_networkx(data, edge_attrs=["edge_weight"])
        G = nx.to_undirected(G)
        G = nx.Graph(G)

        #num_nodes = len(G.nodes())
        #all_nodes = set(G.nodes())
        edges_wo_context_self_loops = list(set(G.edges())- set(zip(context_index.cpu().numpy(), context_index.cpu().numpy())))

        num_edges_wo_context_self_loops = len(edges_wo_context_self_loops)
        #choices = list(itertools.combinations(range(len(edges_wo_context_self_loops)), int(num_edges_wo_context_self_loops * sample_rate)))
        #selected_edge_indexes = list(choices[random.randint(0, len(choices)-1)])
        
        #selected_edge_indexes = random.sample(range(num_edges_wo_context_self_loops), int(num_edges_wo_context_self_loops * sample_rate))

        #removed_edge_indexes = set(range(num_edges_wo_context_self_loops)) - set(selected_edge_indexes)

        removed_edge_indexes = random.sample(range(num_edges_wo_context_self_loops), round(num_edges_wo_context_self_loops * (1-sample_rate)))

        for index in removed_edge_indexes:
            edge = edges_wo_context_self_loops[index]
            G.remove_edge(*edge)

        selected_nodes = [] 
        removed_nodes = []   
        for node in sorted(list(G.nodes())):
            if G.degree(node) == 0:
                removed_nodes.append(node)
            else:
                selected_nodes.append(node)
        
        
        data = pyg_utils.from_networkx(G)

        
        return data.edge_index.to(device), data.edge_weight.to(device), selected_nodes, removed_nodes
        


    def forward(self, x: Tensor, attention_mask: Tensor, edge_index: Adj, edge_weight: Tensor, size: Size = None, **kwargs) -> Tensor:

        # propagate_type: (x: OptPairTensor)
        #print("edge_index.size():", edge_index.size())
        #print("x.size():", x.size())
        
        #x.to(torch.device("cpu"))
        #self.encoder.embeddings.to(torch.device("cpu"))
        
        batch_size, input_seq_len, input_n_embd = x.size()
        
        if self.config.random_walk:
            #print("edge_index before:", edge_index)
            #print("edge_weight before:", edge_weight)
            edge_index, edge_weight, selected_nodes, removed_nodes = self.random_walk(edge_index, edge_weight, kwargs["context_index"], kwargs["random_walk_sample_rate"])
            removed_node_embs = x[removed_nodes]
            
            #print("selected_nodes:", selected_nodes)
            #print("removed_nodes:", removed_nodes)
            #print("edge_index after:", edge_index)
            #print("edge_weight after:", edge_weight)
            
        
        out = self.propagate(edge_index, x=x.view(batch_size, -1), edge_weight=edge_weight, size=size)
        #self.encoder.embeddings.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        #out.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        #print("out.size():", out.size())

        out = out.view(out.size(0), -1, input_n_embd)
        concated_seq_len = out.size(1)
        
        # setup attention_mask
        if attention_mask is not None:
            #print("attention_mask:", attention_mask.size(), attention_mask.cpu().numpy())
            if self.config.aggr == "cat":
                #attention_mask = F.pad(attention_mask, (0,out.size(-2) - attention_mask.size(-1)), "constant", -0.)
                attention_mask = self.propagate(edge_index, x=attention_mask, edge_weight=edge_weight, size=size, pad_value = 1.0)
            
            if self.config.prefix_tuning:
                prefix_attention_mask = torch.ones(batch_size, self.config.pre_seq_len).to(x.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        #print("attention_mask:", list(attention_mask.cpu().numpy()))
        
        # setup past_key_value
        if self.config.prefix_tuning:
            past_key_value = self.create_prefix(batch_size=batch_size)
        else:
            past_key_value = None
        
        # print("out.size():", out.size())
        
        if self.config.name_or_path.startswith("albert"):
            if self.config.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                out = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.layer_module),
                    out, attention_mask)[0]
            else:   
                out = self.layer_module(out, attention_mask = attention_mask)[0]
        elif "deberta" in self.config.name_or_path:
            self.relative_attention = kwargs["relative_attention"]
            self.position_buckets = kwargs["position_buckets"]
            self.max_relative_positions = kwargs["max_relative_positions"]
            relative_pos = self.get_rel_pos(out, None, None)
            #print("relative_pos:", relative_pos.size())
            #print("rel_embeddings:", kwargs["rel_embeddings"].size())
            #print("out:", out.size())
            out = self.layer_module(out, attention_mask = attention_mask, relative_pos=relative_pos, rel_embeddings=kwargs["rel_embeddings"])
        else:
            out = self.layer_module(out, attention_mask = attention_mask, past_key_value=past_key_value)[0]
        
        
        if self.config.random_walk:
            removed_node_embs = F.pad(removed_node_embs, (0,0,0,concated_seq_len - input_seq_len), "constant", 0.0)
            #print("removed_node_embs:", removed_node_embs)
            out = torch.cat([out, removed_node_embs], dim=0)
            out= out[torch.argsort(torch.tensor(selected_nodes + removed_nodes))]

            #print("selected_nodes + removed_nodes:", selected_nodes + removed_nodes)
            #print("torch.argsort(torch.tensor(selected_nodes + removed_nodes)):", torch.argsort(torch.tensor(selected_nodes + removed_nodes)))
        

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if self.config.aggr == "cat":
            return x_j
        else:
            return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        
        
    def aggregate(self, inputs: Tensor, index: Tensor, edge_weight: OptTensor, pad_value: Optional[float] = 0.0,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean", "min", "max" and "mul" operations as
        specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        '''
        print('*'*80)
        print("ptr:", ptr)
        print("inputs.size():", inputs.size())
        print("index:", index.cpu().numpy())
        print("node_dim:", self.node_dim)
        print("dim_size:", dim_size)
        print("self.aggr:", self.aggr)
        print("pad_value:", pad_value)
        print('*'*80)
        '''
        
        if self.config.aggr == "cat":
            max_weight = torch.max(edge_weight)
            edge_weight = edge_weight/max_weight
            
            #seq_len = inputs.size(-1)//self.config.hidden_size
            x = []
            
            for i in range(dim_size):
                x_i = inputs[index == i]
                if len(x_i)>0:
                    x_i = x_i.view(x_i.size(0), self.config.seq_len, -1)
                    edge_weight_i = edge_weight[index == i]
                    
                    x_i = [x_i_k[0] for x_i_k in 
                           sorted([(x_i_k[:,:(x_i_k.size(1)*edge_weight_i[k]).long()], edge_weight_i[k]) 
                                   for k, x_i_k in enumerate(x_i.split(1))], key=lambda x_i_k: x_i_k[1], reverse=True)]
                    x_i = torch.cat(x_i, dim = 1)
                    x.append(x_i)
            max_seq_len = max([x_i.size(1) for x_i in x])
            
            x = [F.pad(x_i, (0,0,0,max_seq_len - x_i.size(1)), "constant", pad_value) for x_i in x]
            x = torch.cat(x, dim=0)
            
            #print("x.size():", x.size())
            return x.view(x.size(0), -1)
        else:
            return super().aggregate(inputs, index, ptr, dim_size) 
            '''
            if ptr is not None:
                ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
                return segment_csr(inputs, ptr, reduce=self.aggr)
            else:
                return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                               reduce=self.aggr)
            '''
    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)