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
from torch_geometric.typing import Adj, OptPairTensor, Size

from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers import AutoConfig
from torch import nn
from transformers import RobertaModel
import torch
from neural_tree.models.prefix_encoder import PrefixEncoder

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



class SAGEFormer2D(MessagePassing):
    
    def __init__(self, encoder, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.encoder = encoder
        #self.encoder.to(device)

        self.normalize = normalize
        
        self.pre_seq_len = self.encoder.config.pre_seq_len
        self.n_layer = self.encoder.config.num_hidden_layers
        self.n_head = self.encoder.config.num_attention_heads
        self.n_embd = self.encoder.config.hidden_size // self.encoder.config.num_attention_heads
        
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_tokens.to(device)
        
        self.prefix_encoder = PrefixEncoder(self.encoder.config)
        self.prefix_encoder.to(device)
        
        self.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob)
        #self.dropout.to(device)
        
    def create_prefix(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def forward(self, x: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:

        # propagate_type: (x: OptPairTensor)
        #print("edge_index.size():", edge_index.size())
        #print("x.size():", x.size())
        
        #x.to(torch.device("cpu"))
        #self.encoder.embeddings.to(torch.device("cpu"))
        out = self.propagate(edge_index, x=x.view(x.size(0), -1), size=size)
        #self.encoder.embeddings.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        #out.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        #print("out.size():", out.size())
        out = out.view(-1, *x.size()[1:])
        
        batch_size = out.shape[0]
        past_key_values = self.create_prefix(batch_size=batch_size)
        
        out = self.encoder.encoder(out, head_mask = [None]*len(self.encoder.encoder.layer), past_key_values=past_key_values)[0]
        

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)