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

from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.models.bert.modeling_bert import BertLayer
from transformers import AutoConfig
from torch import nn
from transformers import RobertaModel
import torch
from neural_tree.models.prefix_encoder import PrefixEncoder
from typing import Optional

'''
class SAGEFormer(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.seq_len = 2
        config = AutoConfig.from_pretrained(
                'roberta-base',
                revision='main',
                hidden_size=self.out_channels // self.seq_len,
                num_attention_heads = 4,
                attention_probs_dropout_prob = 0.5,
                intermediate_size = 16,
            )
        print(config)
        self.former_l = nn.ModuleList([RobertaLayer(config) for _ in range(1)])
        self.former_r = nn.ModuleList([RobertaLayer(config) for _ in range(1)])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        def former_forward(hidden_states, former_layers):
            for i, layer_module in enumerate(former_layers):
                layer_outputs = layer_module(
                        hidden_states,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                hidden_states = layer_outputs[0]
            return hidden_states
        
        
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = former_forward(self.lin_l(out).view(-1, self.seq_len, self.out_channels // self.seq_len), self.former_l).view(-1, self.out_channels)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            #out += self.lin_r(x_r)
            out += former_forward(self.lin_r(x_r).view(-1, self.seq_len, self.out_channels // self.seq_len), self.former_r).view(-1, self.out_channels)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

'''

class SAGEFormer2D(MessagePassing):
    
    def __init__(self, config, normalize: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        
        super().__init__(**kwargs)
        
        self.config = config
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if config.name_or_path.startswith("bert"):
            cls = BertLayer
        elif config.name_or_path.startswith("roberta"):
            cls = RobertaLayer
        else:
            raise RuntimeError("Not implemented")
        
        self.layer_module = cls(config)
        
        if config.prefix_tuning == True:
            for param in self.layer_module.parameters():
                param.requires_grad = False
        
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


    def forward(self, x: Tensor, attention_mask: Tensor, edge_index: Adj, edge_weight: Tensor, 
                size: Size = None) -> Tensor:

        # propagate_type: (x: OptPairTensor)
        #print("edge_index.size():", edge_index.size())
        #print("x.size():", x.size())
        
        #x.to(torch.device("cpu"))
        #self.encoder.embeddings.to(torch.device("cpu"))
        
        batch_size, input_seq_len, input_n_embd = x.size()
        
        
        out = self.propagate(edge_index, x=x.view(batch_size, -1), edge_weight=edge_weight, size=size)
        #self.encoder.embeddings.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        #out.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        #print("out.size():", out.size())

        out = out.view(batch_size, -1, input_n_embd)
        


        # setup attention_mask
        if attention_mask is not None:
            #print("attention_mask:", attention_mask.size(), attention_mask.cpu().numpy())
            if self.config.aggr == "cat":
                #attention_mask = F.pad(attention_mask, (0,out.size(-2) - attention_mask.size(-1)), "constant", -0.)
                attention_mask = self.propagate(edge_index, x=attention_mask, edge_weight=edge_weight, size=size)
            
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
        
        
        out = self.layer_module(out, attention_mask = attention_mask, past_key_value=past_key_value)[0]
        

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
        print(['*']*80)
        print("ptr:", ptr)
        print("inputs.size():", inputs.size())
        print("index:", index.cpu().numpy())
        print("node_dim:", self.node_dim)
        print("dim_size:", dim_size)
        print("self.aggr:", self.aggr)
        print(['*']*80)
        print("pad_value:", pad_value)
        '''
        
        if self.config.aggr == "cat":
            max_weight = torch.max(edge_weight)
            edge_weight = edge_weight/max_weight
            
            #seq_len = inputs.size(-1)//self.config.hidden_size
            x = []
            for i in range(dim_size):
                x_i = inputs[index == i]
                x_i = x_i.view(x_i.size(0), self.config.seq_len, -1)
                edge_weight_i = edge_weight[index == i]
                
                x_i = [x_i_k[:,:(x_i_k.size(1)*edge_weight_i[k]).long()] for k, x_i_k in enumerate(x_i.split(1))]
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