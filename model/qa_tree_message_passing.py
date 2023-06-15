'''
Created on May 9, 2022

@author: Yihao Fang
'''
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from transformers import (
    RobertaModel, MobileBertModel, RobertaTokenizer, 
    RobertaPreTrainedModel, BertModel, BertPreTrainedModel, AlbertPreTrainedModel, AlbertModel,
    PreTrainedModel, PretrainedConfig, DebertaV2PreTrainedModel, DebertaV2Model, DebertaV2Config, AutoTokenizer)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.albert.modeling_albert import AlbertEmbeddings
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Embeddings,build_relative_position
from torch import nn
import torch

from model.sage_former import SAGEFormer2D
from torch_geometric.nn.norm.batch_norm import BatchNorm

from torch.nn import CrossEntropyLoss
from torch_geometric.utils import add_self_loops
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
from transformers.modeling_utils import load_state_dict, no_init_weights, get_checkpoint_shard_files
from transformers.utils import (
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    EntryNotFoundError,
    ModelOutput,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    has_file,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    logging,
    replace_return_docstrings,
)
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from requests import HTTPError

from util.visual_util import animate_graph

from torch.nn import LayerNorm
from mpu import VocabParallelEmbeddings

from mpu.util import divide
from mpu.transformer import PositionalEmbeddings
from mpu.initialize import get_model_parallel_world_size

import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import degree
from torch import Tensor
logger = logging.get_logger(__name__)



def get_qa_tree_network(config):
    if config.name_or_path.startswith("bert"):
        cls = BertPreTrainedModel
    #elif config.name_or_path.startswith("roberta"):
    elif "roberta" in config.name_or_path:
        cls = RobertaPreTrainedModel
    elif config.name_or_path.startswith("albert"):
        cls = AlbertPreTrainedModel
    elif "deberta" in config.name_or_path:
        cls = DebertaV2PreTrainedModel
    elif config.name_or_path.startswith("glm"):
        cls = PreTrainedModel
    else:
        raise RuntimeError("Not implemented")
    class QATreeNetwork(cls):
    #class PrefixNeuralTreeNetwork(BertPreTrainedModel):
        #_keys_to_ignore_on_load_missing = [r"position_ids"]
        
        def __init__(self, config, non_prefix_requires_grad = False):
            """
            NeuralTreeNetwork is the child class of BasicNetwork, which implements basic message passing on graphs.
            The network parameters and loss functions are the same as the parent class. The difference is that this class
             has an additional pooling layer at the end to aggregate final hidden states of the leaf nodes (for node
             classification).
            """
            super().__init__(config)
        
            
            
            #self.roberta = RobertaModel(config, add_pooling_layer=False)
            #self.bert = BertModel(config, add_pooling_layer=False)
            
            if config.name_or_path.startswith("bert"):
                cls = BertEmbeddings
            #elif config.name_or_path.startswith("roberta"):
            elif "roberta" in config.name_or_path:
                cls = RobertaEmbeddings
            elif config.name_or_path.startswith("albert"):
                cls = AlbertEmbeddings
            elif "deberta" in config.name_or_path:
                cls = DebertaV2Embeddings
            elif config.name_or_path.startswith("glm"):
                cls = VocabParallelEmbeddings
            else:
                raise RuntimeError("Not implemented")
            
            self.embeddings = cls(config)
            if config.name_or_path.startswith("albert"):
                self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
                
                
                
            if "deberta" in config.name_or_path:
                self.relative_attention = getattr(config, "relative_attention", False)

                if self.relative_attention:
                    self.max_relative_positions = getattr(config, "max_relative_positions", -1)
                    if self.max_relative_positions < 1:
                        self.max_relative_positions = config.max_position_embeddings
        
                    self.position_buckets = getattr(config, "position_buckets", -1)
                    pos_ebd_size = self.max_relative_positions * 2
        
                    if self.position_buckets > 0:
                        pos_ebd_size = self.position_buckets * 2
        
                    self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)
        
                self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]
        
                if "layer_norm" in self.norm_rel_ebd:
                    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)    
            
            elif config.name_or_path.startswith("glm"):
                self.embedding_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
                self.relative_encoding = config.relative_encoding
                self.block_position_encoding = config.block_position_encoding
                init_method_std=0.02
                layernorm_epsilon=1.0e-5
                if config.relative_encoding:
                    # Relative position embedding
                    self.position_embeddings = PositionalEmbeddings(config.hidden_size)
                    # Per attention head and per partition values.
                    world_size = get_model_parallel_world_size()
                    self.hidden_size_per_attention_head = divide(config.hidden_size,
                                                                 config.num_attention_heads)
                    self.num_attention_heads_per_partition = divide(config.num_attention_heads,
                                                                    world_size)
                    self.r_w_bias = torch.nn.Parameter(
                        torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
                    self.r_w_bias.model_parallel = True
                    self.r_r_bias = torch.nn.Parameter(
                        torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
                    self.r_r_bias.model_parallel = True
                    # Always initialize bias to zero.
                    with torch.no_grad():
                        self.r_w_bias.zero_()
                        self.r_r_bias.zero_()
                else:
                    raise RuntimeError("Not implemented")
                    '''
                    # Position embedding (serial).
                    if config.block_position_encoding:
                        self.position_embeddings = torch.nn.Embedding(config.seq_len + 1, config.hidden_size)
                        self.block_position_embeddings = torch.nn.Embedding(config.seq_len + 1, config.hidden_size)
                        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
                    else:
                        self.position_embeddings = torch.nn.Embedding(config.seq_len, config.hidden_size)
                    # Initialize the position embeddings.
                    torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)   
                    '''
                self.final_layernorm = LayerNorm(config.hidden_size, eps=layernorm_epsilon)
                
            
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            self.classifier = torch.nn.Linear(config.hidden_size, 1)
            
            
            
            if self.config.prefix_tuning == True:
                for param in self.embeddings.parameters():
                    param.requires_grad = non_prefix_requires_grad
            '''
            for param in self.embeddings.parameters():
                param.requires_grad = False
            '''

            
            # message passing
            self.convs = nn.ModuleList()
            if config.name_or_path.startswith("albert"):
                self.convs.append(SAGEFormer2D(config, normalize=False))
            else:
                for i in range(config.num_hidden_layers):
                    '''
                    if i > config.num_hidden_layers -3:
                        non_prefix_requires_grad = True
                    else: 
                        non_prefix_requires_grad = False
                    '''
                    self.convs.append(SAGEFormer2D(config, normalize=False, non_prefix_requires_grad=non_prefix_requires_grad))
            
            

            self.init_weights()

        
        def unbatch_edge(self, edge_index: Tensor, edge_weight: Tensor, batch: Tensor):
            r"""
            Args:
                batch (LongTensor): The batch vector
                    :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
                    node to a specific example. Must be ordered.
        
            :rtype: :class:`List[Tensor]`
            """
            deg = degree(batch, dtype=torch.int64)
            ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)
        
            edge_batch = batch[edge_index[0]]
            
            #print("edge_batch:", edge_batch)
            
            edge_index = edge_index - ptr[edge_batch]
            
            num_batches = torch.max(batch)+1
            
            edge_batch_l = []
            edge_weight_l = []
            for i in range(num_batches):
                edge_batch_i_mask = (edge_batch == i)
                edge_batch_l.append(edge_index[:, edge_batch_i_mask])
                edge_weight_l.append(edge_weight[edge_batch_i_mask])
            
            return edge_batch_l, edge_weight_l
        
        def augment_graph(self, x, attention_mask, edge_index, edge_weight, node_context_mask, leaf_mask, y, batch):
            device = x.device
            
            x_l_1 = pyg_utils.unbatch(x, batch)
            attention_mask_l_1 = pyg_utils.unbatch(attention_mask, batch)
            edge_index_l_1, edge_weight_l_1 = self.unbatch_edge(edge_index, edge_weight, batch)
            #edge_weight_l_1 = edge_weight.split([ei.size(1) for ei in edge_index_l_1])
            #print("edge_index_l_1:", edge_index_l_1)
            #print("edge_weight_l_1:", edge_weight_l_1)
            
            node_context_mask_l_1 = pyg_utils.unbatch(node_context_mask, batch)
            leaf_mask_l_1 = pyg_utils.unbatch(leaf_mask, batch)
            y_l_1 = y.split(1)
            
            x_l_2=[]
            attention_mask_l_2=[]
            edge_index_l_2=[]
            edge_weight_l_2=[]
            node_context_mask_l_2=[]
            leaf_mask_l_2=[]
            y_l_2=[]
            context_index_l_2=[]
            ground_true_mask_l_2=[]
            positive_mask_l_2=[]
            negative_mask_l_2=[]
            batch_l_2=[]
            cum_index = 0
            
            for i, (x, attention_mask, edge_index, edge_weight, node_context_mask, leaf_mask, y) in enumerate(zip(x_l_1, attention_mask_l_1, edge_index_l_1, edge_weight_l_1, node_context_mask_l_1, leaf_mask_l_1, y_l_1)):
            
                x_index = torch.arange(x.size(0)).long()
                context_index = x_index[(node_context_mask>=0) & leaf_mask]
            
                data = Data(edge_index = edge_index, edge_weight = edge_weight)
                G = pyg_utils.to_networkx(data, edge_attrs=["edge_weight"])
                G = nx.to_undirected(G)
                G = nx.Graph(G)
                
                context_value = node_context_mask[context_index]
                mask_value_index = torch.arange(context_value.size(0)).long()
                ground_true_mask_value_index = mask_value_index[context_value == y]
                ground_true_context_index = context_index[ground_true_mask_value_index].cpu().numpy()
                
                for connected_component in nx.connected_components(G):
                    if ground_true_context_index[0] in connected_component:
                        ground_true_subgraph_index = sorted(connected_component)
                        sub_G = G.subgraph(connected_component)
                        break
                
                not_aug_size = x.size(0)
                aug_size = len(ground_true_subgraph_index)
                
                x = torch.cat([x, x[ground_true_subgraph_index]])
                attention_mask = torch.cat([attention_mask, attention_mask[ground_true_subgraph_index]])
                
                node_context_mask = torch.cat([node_context_mask, node_context_mask[ground_true_subgraph_index]])
                leaf_mask = torch.cat([leaf_mask, leaf_mask[ground_true_subgraph_index]])
                
                x_index = torch.arange(x.size(0)).long()
                context_index = x_index[(node_context_mask>=0) & leaf_mask]
                
                #mapping = dict(zip(sorted(sub_G), range(not_aug_size,len(sub_G)+not_aug_size)))
                #sub_G = nx.relabel_nodes(sub_G, mapping)
                sub_G_data = pyg_utils.from_networkx(sub_G)
                
                
                edge_index = torch.cat([edge_index, sub_G_data.edge_index.to(device)+not_aug_size], dim = 1)
                edge_weight = torch.cat([edge_weight, sub_G_data.edge_weight.to(device)])
                
                
                ground_true_mask = torch.zeros(not_aug_size + aug_size).bool().to(device)
                ground_true_mask[ground_true_subgraph_index] = True
                
                positive_mask = torch.cat([torch.zeros(not_aug_size).bool(), torch.ones(aug_size).bool()]).to(device)
    
                negative_mask = torch.cat([torch.ones(not_aug_size).bool(), torch.zeros(aug_size).bool()]).to(device)
                negative_mask[ground_true_subgraph_index] = False
                
                x_l_2.append(x)
                attention_mask_l_2.append(attention_mask)
                edge_index_l_2.append(edge_index+cum_index)
                edge_weight_l_2.append(edge_weight)
                node_context_mask_l_2.append(node_context_mask)
                leaf_mask_l_2.append(leaf_mask)
                y_l_2.append(y)
                context_index_l_2.append(context_index+cum_index)
                ground_true_mask_l_2.append(ground_true_mask)
                positive_mask_l_2.append(positive_mask)
                negative_mask_l_2.append(negative_mask)
                batch_l_2.append(torch.ones(x.size(0),dtype=torch.int64).to(device)*i)
                cum_index += x.size(0)
                
            
            return torch.cat(x_l_2), torch.cat(attention_mask_l_2), torch.cat(edge_index_l_2, dim=1), torch.cat(edge_weight_l_2),\
                 torch.cat(context_index_l_2), torch.cat(node_context_mask_l_2), torch.cat(leaf_mask_l_2), \
                 torch.cat(ground_true_mask_l_2), torch.cat(positive_mask_l_2), torch.cat(negative_mask_l_2), torch.cat(batch_l_2)

        def global_mean_pool(self, x, batch, num_choices, choice_mask, pool_mask, squeeze = False):
            batch = batch[pool_mask]*num_choices+choice_mask[pool_mask].long()
            if squeeze:
                while torch.max(batch) >= 0:
                    temp = batch[batch>=0]
                    batch[batch>=0] = temp - torch.min(temp)
                    batch = batch - 1
                batch = batch - torch.min(batch)    
            return pyg_nn.global_mean_pool(x[pool_mask, :], batch)
    
    
        def isotropy_loss(self, embeddings):
            # covariance
            meanVector = embeddings.mean(dim=0)
            centereVectors = embeddings - meanVector
    
            # estimate covariance matrix
            featureDim = meanVector.shape[0]
            dataCount = embeddings.shape[0]
            covMatrix = ((centereVectors.t())@centereVectors)/(dataCount-1) 
    
            # normalize covariance matrix
            stdVector = torch.std(embeddings, dim=0)
            sigmaSigmaMatrix = (stdVector.unsqueeze(1))@(stdVector.unsqueeze(0))
            normalizedConvMatrix = covMatrix/sigmaSigmaMatrix
    
            deltaMatrix = normalizedConvMatrix - torch.eye(featureDim).to(self.device)
    
            covLoss = torch.norm(deltaMatrix)   # Frobenius norm
            
            return covLoss
    
        def forward(self, data):

            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            x, token_type_ids, attention_mask, edge_index, batch = data.x, data.node_token_type_ids, data.node_attention_mask, data.edge_index, data.batch
            node_context_mask, y = data.node_context_mask, data.y
            if hasattr(data,'leaf_mask'):
                leaf_mask = data.leaf_mask
            else:
                leaf_mask = torch.ones_like(node_context_mask, dtype=bool)
            #print("qid:", data.qid)
            
            num_choices = torch.max(node_context_mask[leaf_mask]).long().item() + 1
            
            if data.num_node_features == 0:
                raise RuntimeError('No node feature')
            
            if self.config.prefix_tuning:
                past_key_values_length = self.config.pre_seq_len
            else:
                past_key_values_length = 0
            
            
            
            # setup input embeddings
            x_all_len, seq_len = x.size()
            
            x_index = torch.arange(x.size(0)).long()

            context_index = x_index[(node_context_mask>=0) & leaf_mask]
            x_index = torch.cat([x_index[leaf_mask], x_index[torch.logical_not(leaf_mask)]])
            x_index = torch.argsort (x_index)
            
            if "deberta" in config.name_or_path:
                x = self.embeddings(
                    input_ids=x[leaf_mask].long(),
                    token_type_ids=token_type_ids[leaf_mask].long(),
                    mask=attention_mask[leaf_mask].long()
                )
            else:
                x = self.embeddings(
                    input_ids=x[leaf_mask].long(),
                    token_type_ids=token_type_ids[leaf_mask].long(),
                    past_key_values_length = past_key_values_length
                )
                
            x_leaf_len = x.size(0)
            
            x = torch.cat([x, torch.zeros(x_all_len-x_leaf_len, *x.size()[1:]).to(x.device)])[x_index]
            
            if self.config.name_or_path.startswith("albert"):
                x = self.embedding_hidden_mapping_in(x)
                
                
            if "deberta" in config.name_or_path:    
                rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
                if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
                    rel_embeddings = self.LayerNorm(rel_embeddings)
                

            
            if config.name_or_path.startswith("glm"):
                if self.relative_encoding:
                    position_sequence = torch.arange(seq_len - 1, -1, -1.0, device=x.device,
                                                     dtype=x.dtype)
                    position_embeddings = self.position_embeddings(position_sequence)
                    # Apply dropout
                    position_embeddings = self.embedding_dropout(position_embeddings)
                
                else:
                    raise RuntimeError("Not implemented")
                    '''
                    if self.block_position_encoding:
                        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
                    position_embeddings = self.position_embeddings(position_ids)
                    x = x + position_embeddings
                    if self.block_position_encoding:
                        block_position_embeddings = self.block_position_embeddings(block_position_ids)
                        x = x + block_position_embeddings
                    '''
                x = self.embedding_dropout(x)  
            
            
            x_n_embd = x.size(2)
            
            
            # setup attention mask
            attention_mask = torch.cat([attention_mask[leaf_mask], \
                    torch.max(attention_mask[leaf_mask], dim=0, keepdim=True)[0].expand(x_all_len-x_leaf_len, -1)])[x_index]
            
            #attention_mask = torch.max(attention_mask[leaf_mask], dim=0, keepdim=True)[0].expand(x_all_len-x_leaf_len, -1)
            
            #attention_mask = self.get_extended_attention_mask(attention_mask, x.size()[:-1], x.device)

            
            # setup edge weight and add self loops
            edge_weight = torch.mul(torch.ones(edge_index.size(1)).to(edge_index.device), 0.2) #0.2
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, 1.0, x_all_len) #1.0
            
            if self.config.aggr == "cat":
                non_self_loop_mask = edge_index[0] != edge_index[1]
                cut_index = torch.sum(non_self_loop_mask.long())
                edge_index = torch.cat([edge_index[:,cut_index:], edge_index[:,:cut_index]], dim=1)
                edge_weight = torch.cat([edge_weight[cut_index:], edge_weight[:cut_index]], dim=0)
            
            def equal(input, other):
                rt = torch.zeros_like(input).bool()
                for item in other:
                    rt = torch.logical_or(input==item, rt)
                    #print("rt:", rt.size(), list(rt.cpu().numpy()))
                return rt
            #print("context_index:", context_index.size(), list(context_index.cpu().numpy()))
            context_self_loop_mask = torch.logical_and(equal(edge_index[0], context_index), equal(edge_index[1], context_index))
            edge_weight[context_self_loop_mask] = 2.0 # 2.0
            

            if self.config.contrastive_loss and self.training:
                #print("edge_index before:", edge_index)
                #print("edge_weight before:", edge_weight)
                #print("node_context_mask before:", node_context_mask)
                #print("leaf_mask before:", leaf_mask)
                #print("context_index before:", context_index)
                #print("batch before:", batch)
                x, attention_mask, edge_index, edge_weight, context_index, node_context_mask, leaf_mask, ground_true_mask, positive_mask, negative_mask, batch = self.augment_graph(x, attention_mask, edge_index, edge_weight, node_context_mask, leaf_mask, y, batch)
                #print("edge_index after:", edge_index)
                #print("edge_weight after:", edge_weight)
                #print("node_context_mask after:", node_context_mask)
                #print("leaf_mask after:", leaf_mask)
                #print("context_index after:", context_index)
                #print("batch after:", batch)
                
            if config.visualize:
                node_hidden_states = []
                node_hidden_states.append(x.detach().cpu().numpy())
            
            for i in range(int(self.config.num_hidden_layers * self.config.hidden_layer_retention_rate)):
                attention_mask = attention_mask if i < self.config.first_attn_mask_layers else None
                
                '''
                if i in [2,3,4]:
                    x = x.cuda(1)
                    attention_mask = attention_mask.cuda(1)
                    edge_index = edge_index.cuda(1)
                    edge_weight = edge_weight.cuda(1)
                    for param in self.convs[i].parameters():
                        param = param.cuda(1)
                '''
                if self.config.name_or_path.startswith("albert"):
                    idx = 0
                else:
                    idx = i
                
                if "deberta" in config.name_or_path:
                    #print("x.size():", x.size())
                    x = self.convs[idx](x, attention_mask, 
                                        relative_attention = self.relative_attention, 
                                        position_buckets = self.position_buckets, 
                                        max_relative_positions = self.max_relative_positions,
                                        rel_embeddings = rel_embeddings, 
                                        edge_index = edge_index, edge_weight = edge_weight, context_index = context_index, random_walk_sample_rate = self.config.random_walk_sample_rate)
                elif config.name_or_path.startswith("glm"):
                    args = [x, attention_mask]
                    if self.relative_encoding:
                        args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                    x = self.convs[idx](*args, edge_index = edge_index, edge_weight = edge_weight, context_index = context_index, random_walk_sample_rate = self.config.random_walk_sample_rate)
                else:
                    x = self.convs[idx](x, attention_mask, edge_index = edge_index, edge_weight = edge_weight, context_index = context_index, random_walk_sample_rate = self.config.random_walk_sample_rate)
                
                
                if self.config.aggr == "cat":
                    x = x[:,:seq_len]
                #print("x:", x.cpu().detach().numpy())
                '''
                if i != len(self.convs) - 1:    # activation and dropout, except for the last iteration
                    if self.conv_block == 'GIN':
                        x = self.batch_norms[i](x)
                    x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                '''
                if config.visualize:
                    node_hidden_states.append(x.detach().cpu().numpy())
            
            
                    
            if config.visualize:
                animate_graph(model_name=config._name_or_path, 
                                node_hidden_states=node_hidden_states, 
                                embeddings=self.embeddings.word_embeddings, 
                                embedding_projection=self.embedding_hidden_mapping_in,
                                node_context_mask=node_context_mask.long().cpu().numpy(), 
                                leaf_mask=leaf_mask.cpu().numpy(), 
                                batch_mask=batch.long().cpu().numpy(),
                                edge_index=edge_index.cpu().numpy(),
                                qid=data.qid)
                
                
            if config.name_or_path.startswith("glm"):          
                x = self.final_layernorm(x)        
            
            #x = self.roberta.pooler(x)
            x = x[:,0]
            
            
            if self.config.graph_pooling == "leaf":
                pool_mask = leaf_mask
                raise RuntimeError("Choice mask not implemented")
            elif self.config.graph_pooling == "context":
                pool_mask = (node_context_mask>=0) & leaf_mask
                choice_mask = node_context_mask
            elif self.config.graph_pooling == "all":
                pool_mask = torch.ones(x.size(0)).bool()
                raise RuntimeError("Choice mask not implemented")
            else:
                raise RuntimeError("Not implemented")
            
            
            if self.config.contrastive_loss and self.training:
                pool_ground_true_mask = pool_mask & ground_true_mask
                pool_positive_mask = pool_mask & positive_mask  #augmentation
                pool_negative_mask = pool_mask & negative_mask
                
                ground_true_x = self.global_mean_pool(x, batch, num_choices, choice_mask, pool_ground_true_mask, squeeze=True)
                positive_x = self.global_mean_pool(x, batch, num_choices, choice_mask, pool_positive_mask, squeeze=True)
                negative_x = self.global_mean_pool(x, batch, num_choices, choice_mask, pool_negative_mask, squeeze=True).view(-1, num_choices-1, x_n_embd)
                
                sim = torch.nn.CosineSimilarity(dim=1)
                negative_sim = torch.sum(torch.stack([torch.exp(sim(ground_true_x, torch.squeeze(negative_x_per_choice)))
                                for negative_x_per_choice in negative_x.split(1, dim = 1)] 
                                                      + [torch.exp(sim(torch.squeeze(negative_x_per_choice), ground_true_x))
                                                         for negative_x_per_choice in negative_x.split(1, dim = 1)], dim=1), dim=1)
                positive_sim = torch.exp(sim(ground_true_x, positive_x))
                
                contrastive_loss = -torch.mean(torch.log(positive_sim/(positive_sim+negative_sim)))
                
                #print("positive_sim:", positive_sim.detach().cpu().numpy())
                #print("negative_sim:", negative_sim.detach().cpu().numpy())
                
                iso_x = self.global_mean_pool(x, batch, num_choices, choice_mask, pool_mask & (~positive_mask))
                isotropy_loss = self.isotropy_loss(iso_x)
                
                
                
                ce_x = self.global_mean_pool(x, batch, num_choices, choice_mask, pool_mask & (~positive_mask))
    
                ce_x = self.classifier(ce_x)
                ce_x = ce_x.view(-1, num_choices)
                cross_entropy_loss_fct = CrossEntropyLoss(reduction='mean')
                cross_entropy_loss = cross_entropy_loss_fct(ce_x, y)
                
                #print("cross_entropy_loss:", cross_entropy_loss.item())
                #print("contrastive_loss:", contrastive_loss.item())
                #print("isotropy_loss:", isotropy_loss.item())
                #print("self.config.cross_entropy_loss_scalar:", self.config.cross_entropy_loss_scalar)
                #print("self.config.contrastive_loss_scalar:", self.config.contrastive_loss_scalar)
                #print("self.config.isotropy_loss_scalar:", self.config.isotropy_loss_scalar)
                
                
                loss = self.config.cross_entropy_loss_scalar * cross_entropy_loss + self.config.contrastive_loss_scalar * contrastive_loss + self.config.isotropy_loss_scalar * isotropy_loss
                
                #print("loss:", loss.item())
                
            else:
                x = self.global_mean_pool(x, batch, num_choices, choice_mask, pool_mask)
                #x = pyg_nn.global_mean_pool(x[pool_mask, :], batch[pool_mask]*num_choices+node_context_mask[pool_mask].long())
    
                x = self.classifier(x)
                x = x.view(-1, num_choices)
                cross_entropy_loss_fct = CrossEntropyLoss(reduction='mean')
                loss = cross_entropy_loss_fct(x, y)
            return x, loss

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
            r"""
            Instantiate a pretrained pytorch model from a pre-trained model configuration.
    
            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
            the model, you should first set it back in training mode with `model.train()`.
    
            The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
            pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
            task.
    
            The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
            weights are discarded.
    
            Parameters:
                pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                    Can be either:
    
                        - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                          Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                          user or organization name, like `dbmdz/bert-base-german-cased`.
                        - A path to a *directory* containing model weights saved using
                          [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                        - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                          this case, `from_tf` should be set to `True` and a configuration object should be provided as
                          `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                          PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                        - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                          `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                          `True`.
                        - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                          arguments `config` and `state_dict`).
                model_args (sequence of positional arguments, *optional*):
                    All remaining positional arguments will be passed to the underlying model's `__init__` method.
                config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                    Can be either:
    
                        - an instance of a class derived from [`PretrainedConfig`],
                        - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].
    
                    Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                    be automatically loaded when:
    
                        - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                          model).
                        - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                          save directory.
                        - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                          configuration JSON file named *config.json* is found in the directory.
                state_dict (`Dict[str, torch.Tensor]`, *optional*):
                    A state dictionary to use instead of a state dictionary loaded from saved weights file.
    
                    This option can be used if you want to create a model from a pretrained configuration but load your own
                    weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                    [`~PreTrainedModel.from_pretrained`] is not a simpler option.
                cache_dir (`Union[str, os.PathLike]`, *optional*):
                    Path to a directory in which a downloaded pretrained model configuration should be cached if the
                    standard cache should not be used.
                from_tf (`bool`, *optional*, defaults to `False`):
                    Load the model weights from a TensorFlow checkpoint save file (see docstring of
                    `pretrained_model_name_or_path` argument).
                from_flax (`bool`, *optional*, defaults to `False`):
                    Load the model weights from a Flax checkpoint save file (see docstring of
                    `pretrained_model_name_or_path` argument).
                ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                    Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                    as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                    checkpoint with 3 labels).
                force_download (`bool`, *optional*, defaults to `False`):
                    Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                    cached versions if they exist.
                resume_download (`bool`, *optional*, defaults to `False`):
                    Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                    file exists.
                proxies (`Dict[str, str]`, *optional*):
                    A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                    'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
                output_loading_info(`bool`, *optional*, defaults to `False`):
                    Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
                local_files_only(`bool`, *optional*, defaults to `False`):
                    Whether or not to only look at local files (i.e., do not try to download the model).
                use_auth_token (`str` or *bool*, *optional*):
                    The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                    when running `transformers-cli login` (stored in `~/.huggingface`).
                revision (`str`, *optional*, defaults to `"main"`):
                    The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                    git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                    identifier allowed by git.
                mirror (`str`, *optional*):
                    Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                    problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                    Please refer to the mirror site for more information.
                _fast_init(`bool`, *optional*, defaults to `True`):
                    Whether or not to disable fast initialization.
                low_cpu_mem_usage(`bool`, *optional*, defaults to `False`):
                    Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                    This is an experimental feature and a subject to change at any moment.
                torch_dtype (`str` or `torch.dtype`, *optional*):
                    Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                    will be automatically derived from the model's weights.
    
                    <Tip warning={true}>
    
                    One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                    4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                    [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.
    
                    </Tip>
    
                kwargs (remaining dictionary of keyword arguments, *optional*):
                    Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                    `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                    automatically loaded:
    
                        - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                          underlying model's `__init__` method (we assume all relevant updates to the configuration have
                          already been done)
                        - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                          initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                          corresponds to a configuration attribute will be used to override said attribute with the
                          supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                          will be passed to the underlying model's `__init__` function.
    
            <Tip>
    
            Passing `use_auth_token=True`` is required when you want to use a private model.
    
            </Tip>
    
            <Tip>
    
            Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
            use this method in a firewalled environment.
    
            </Tip>
    
            Examples:
    
            ```python
            >>> from transformers import BertConfig, BertModel
    
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = BertModel.from_pretrained("bert-base-uncased")
            >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
            >>> model = BertModel.from_pretrained("./test/saved_model/")
            >>> # Update configuration during loading.
            >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
            >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
            >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
            >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
            ```"""
            config = kwargs.pop("config", None)
            state_dict = kwargs.pop("state_dict", None)
            cache_dir = kwargs.pop("cache_dir", None)
            from_tf = kwargs.pop("from_tf", False)
            from_flax = kwargs.pop("from_flax", False)
            ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
            force_download = kwargs.pop("force_download", False)
            resume_download = kwargs.pop("resume_download", False)
            proxies = kwargs.pop("proxies", None)
            output_loading_info = kwargs.pop("output_loading_info", False)
            local_files_only = kwargs.pop("local_files_only", False)
            use_auth_token = kwargs.pop("use_auth_token", None)
            revision = kwargs.pop("revision", None)
            mirror = kwargs.pop("mirror", None)
            from_pipeline = kwargs.pop("_from_pipeline", None)
            from_auto_class = kwargs.pop("_from_auto", False)
            _fast_init = kwargs.pop("_fast_init", True)
            torch_dtype = kwargs.pop("torch_dtype", None)
            low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
    
            from_pt = not (from_tf | from_flax)
    
            user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
            if from_pipeline is not None:
                user_agent["using_pipeline"] = from_pipeline
    
            if is_offline_mode() and not local_files_only:
                logger.info("Offline mode: forcing local_files_only=True")
                local_files_only = True
                
                
            if config.name_or_path.startswith("glm"):
                model = cls(config)
                # make sure token embedding weights are still tied if needed
                model.tie_weights()
        
                # Set model in evaluation mode to deactivate DropOut modules by default
                model.eval()
                return model
    
            # Load config if we don't provide a configuration
            if not isinstance(config, PretrainedConfig):
                config_path = config if config is not None else pretrained_model_name_or_path
                config, model_kwargs = cls.config_class.from_pretrained(
                    config_path,
                    cache_dir=cache_dir,
                    return_unused_kwargs=True,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            else:
                model_kwargs = kwargs
    
            # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
            # index of the files.
            is_sharded = False
            sharded_metadata = None
            # Load model
            if pretrained_model_name_or_path is not None:
                pretrained_model_name_or_path = str(pretrained_model_name_or_path)
                if os.path.isdir(pretrained_model_name_or_path):
                    if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                        # Load from a TF 1.0 checkpoint in priority if from_tf
                        archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                    elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                        # Load from a TF 2.0 checkpoint in priority if from_tf
                        archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                    elif from_flax and os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                        # Load from a Flax checkpoint in priority if from_flax
                        archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                    elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                        # Load from a PyTorch checkpoint
                        archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                    elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                        # Load from a sharded PyTorch checkpoint
                        archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                        is_sharded = True
                    # At this stage we don't have a weight file so we will raise an error.
                    elif os.path.isfile(
                        os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                    ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                        raise EnvironmentError(
                            f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                            "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                            "weights."
                        )
                    elif os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME):
                        raise EnvironmentError(
                            f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                            "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                            "weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
                            f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                        )
                elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                    archive_file = pretrained_model_name_or_path
                elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                    if not from_tf:
                        raise ValueError(
                            f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                            "from_tf to True to load from this checkpoint."
                        )
                    archive_file = pretrained_model_name_or_path + ".index"
                else:
                    # set correct filename
                    if from_tf:
                        filename = TF2_WEIGHTS_NAME
                    elif from_flax:
                        filename = FLAX_WEIGHTS_NAME
                    else:
                        filename = WEIGHTS_NAME
    
                    archive_file = hf_bucket_url(
                        pretrained_model_name_or_path,
                        filename=filename,
                        revision=revision,
                        mirror=mirror,
                    )
    
                try:
                    # Load from URL or cache if already cached
                    resolved_archive_file = cached_path(
                        archive_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
    
                except RepositoryNotFoundError:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                        "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                        "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                        "login` and pass `use_auth_token=True`."
                    )
                except RevisionNotFoundError:
                    raise EnvironmentError(
                        f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                        "this model name. Check the model page at "
                        f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                    )
                except EntryNotFoundError:
                    if filename == WEIGHTS_NAME:
                        try:
                            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                            archive_file = hf_bucket_url(
                                pretrained_model_name_or_path,
                                filename=WEIGHTS_INDEX_NAME,
                                revision=revision,
                                mirror=mirror,
                            )
                            resolved_archive_file = cached_path(
                                archive_file,
                                cache_dir=cache_dir,
                                force_download=force_download,
                                proxies=proxies,
                                resume_download=resume_download,
                                local_files_only=local_files_only,
                                use_auth_token=use_auth_token,
                                user_agent=user_agent,
                            )
                            is_sharded = True
                        except EntryNotFoundError:
                            # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                            # message.
                            has_file_kwargs = {
                                "revision": revision,
                                "mirror": mirror,
                                "proxies": proxies,
                                "use_auth_token": use_auth_token,
                            }
                            if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                    "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                                    "weights."
                                )
                            elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                    "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                                    "weights."
                                )
                            else:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}, "
                                    f"{TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                        )
                except HTTPError as err:
                    raise EnvironmentError(
                        f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                        f"{err}"
                    )
                except ValueError:
                    raise EnvironmentError(
                        f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
                        f"files and it looks like {pretrained_model_name_or_path} is not the path to a directory "
                        f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                        f"{FLAX_WEIGHTS_NAME}.\n"
                        "Checkout your internet connection or see how to run the library in offline mode at "
                        "'https://huggingface.co/docs/transformers/installation#offline-mode'."
                    )
                except EnvironmentError:
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                        "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                        f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                        f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                        f"{FLAX_WEIGHTS_NAME}."
                    )
    
                if resolved_archive_file == archive_file:
                    logger.info(f"loading weights file {archive_file}")
                else:
                    logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
            else:
                resolved_archive_file = None
    
            # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
            if is_sharded:
                # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
                resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    resolved_archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    revision=revision,
                    mirror=mirror,
                )
            # load pt weights early so that we know which dtype to init the model under
            if from_pt:
                if not is_sharded:
                    # Time to load the checkpoint
                    state_dict = load_state_dict(resolved_archive_file)
                # set dtype to instantiate the model under:
                # 1. If torch_dtype is not None, we use that dtype
                # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
                #    weights entry - we assume all weights are of the same dtype
                # we also may have config.torch_dtype available, but we won't rely on it till v5
                dtype_orig = None
                if torch_dtype is not None:
                    if isinstance(torch_dtype, str):
                        if torch_dtype == "auto":
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = next(iter(state_dict.values())).dtype
                            else:
                                one_state_dict = load_state_dict(resolved_archive_file)
                                torch_dtype = next(iter(one_state_dict.values())).dtype
                                del one_state_dict  # free CPU memory
                        else:
                            raise ValueError(
                                f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                            )
                    dtype_orig = cls._set_default_torch_dtype(torch_dtype)
    
                if low_cpu_mem_usage:
                    # save the keys
                    if is_sharded:
                        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
                    else:
                        loaded_state_dict_keys = [k for k in state_dict.keys()]
                        del state_dict  # free CPU memory - will reload again later
    
            #print("state_dict.keys():", state_dict.keys())
            
            if config.name_or_path.startswith("bert"):
                prefix = "bert"
                layer = "layer"
            #elif config.name_or_path.startswith("roberta"):
            elif "roberta" in config.name_or_path:
                prefix = "roberta"
                layer = "layer"
            elif config.name_or_path.startswith("albert"):
                prefix = "albert"
                layer = "albert_layer_groups.0.albert_layers"
            elif "deberta" in config.name_or_path:
                prefix = "deberta"
                layer = "layer"
            else:
                raise RuntimeError("Not implemented")
            
            
            
            import re
            key_pairs = []
            for key in state_dict:
                new_key = re.sub(prefix+r".encoder."+layer+r".([0-9]+)", prefix+r".convs.\1.layer_module", key)
                if config.name_or_path.startswith("albert"):
                    new_key = re.sub("albert.encoder.embedding_hidden_mapping_in", "albert.embedding_hidden_mapping_in", new_key)
                elif "deberta" in config.name_or_path:
                    new_key = re.sub("deberta.encoder.rel_embeddings.weight", "deberta.rel_embeddings.weight", new_key)
                    new_key = re.sub("deberta.encoder.LayerNorm.bias", "deberta.LayerNorm.bias", new_key)
                    new_key = re.sub("deberta.encoder.LayerNorm.weight", "deberta.LayerNorm.weight", new_key)
                key_pairs.append((key, new_key))
            for old_key, new_key in key_pairs:
                state_dict[new_key] = state_dict.pop(old_key)
                
            #print("state_dict.keys():", state_dict.keys())
    
            config.name_or_path = pretrained_model_name_or_path
    
            # Instantiate model.
            if is_deepspeed_zero3_enabled():
                import deepspeed
    
                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                # this immediately partitions the model across all gpus, to avoid the overhead in time
                # and memory copying it on CPU or each GPU first
                with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                    with no_init_weights(_enable=_fast_init):
                        model = cls(config, *model_args, **model_kwargs)
            else:
                with no_init_weights(_enable=_fast_init):
                    model = cls(config, *model_args, **model_kwargs)
    
            if from_pt:
                # restore default dtype
                if dtype_orig is not None:
                    torch.set_default_dtype(dtype_orig)
    
            if from_tf:
                if resolved_archive_file.endswith(".index"):
                    # Load from a TensorFlow 1.X checkpoint - provided by original authors
                    model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
                else:
                    # Load from our TensorFlow 2.0 checkpoints
                    try:
                        from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model
    
                        model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                    except ImportError:
                        logger.error(
                            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                        )
                        raise
            elif from_flax:
                try:
                    from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
    
                    model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
                except ImportError:
                    logger.error(
                        "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see "
                        "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
                    )
                    raise
            elif from_pt:
    
                if low_cpu_mem_usage:
                    cls._load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file)
                else:
                    model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                        model,
                        state_dict,
                        resolved_archive_file,
                        pretrained_model_name_or_path,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        sharded_metadata=sharded_metadata,
                        _fast_init=_fast_init,
                    )
    
            # make sure token embedding weights are still tied if needed
            model.tie_weights()
    
            # Set model in evaluation mode to deactivate DropOut modules by default
            model.eval()
    
            if output_loading_info:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
                return model, loading_info
    
            return model
      
    return QATreeNetwork 
        