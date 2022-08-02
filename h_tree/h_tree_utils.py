import networkx as nx
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from copy import deepcopy
from h_tree import generate_jth, generate_node_labels, sample_and_generate_jth
import os
import concurrent.futures
from collections import defaultdict
from typing import List, Optional, Tuple, Union
import torch_geometric.data

class HTreeDataset:
    """
    H-Tree dataset
        data_list: a list of torch_geometric.data.Data instances (graph classification) or a list of three such lists
         corresponding to train, val, test split (node classification)
        num_node_features:  int
        num_classes:        int
        name:               string
        task:               string (node or graph)
    """
    def __init__(self, data_list, num_node_features, num_classes, name, task, data_list_original=None):
        assert isinstance(num_node_features, int)
        if data_list_original is not None:
            assert isinstance(data_list_original[0], Data)
        assert isinstance(num_classes, int) or isinstance(num_classes, tuple)
        assert task == 'graph' or task == 'node'
        if task == 'graph':
            assert isinstance(data_list[0], Data)
        else:
            assert len(data_list) == 3
            for i in range(3):
                assert len(data_list) == 0 or isinstance(data_list[i][0], Data)

        self.dataset_jth = data_list
        self.dataset_original = data_list_original
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.name = name
        self.task = task



def from_networkx(G, group_node_attrs: Optional[Union[List[str], all]] = None,
                  group_edge_attrs: Optional[Union[List[str], all]] = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(G.edges(keys=False))
    else:
        edges = list(G.edges)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in data.items():
        try:
            data[key] = torch.tensor(value)
        except ValueError:
            pass
        except TypeError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def convert_to_networkx_jth(data: Data, task='graph', node_id=None, radius=None, node_attrs=None, edge_attrs=None, treewidth = 1):
    """
    Convert a graph or its subgraph given id and radius (an ego graph) to junction tree hierarchy. The node features in
    the input graph will be copied to the corresponding leaf nodes of the output tree decomposition.
    :param data: torch_geometric.data.Data, input graph
    :param task: 'node' or 'graph'
    :param node_id: int, node id in the input graph to be classified (only used when task='node')
    :param radius: int, radius of the ego graph around classification node to be converted (only used when task='node')
    :returns: data_jth, G_jth, root_nodes
    """
    # Convert to networkx graph
    if node_attrs is None:
        node_attrs=['x']
    G = pyg_utils.to_networkx(data, node_attrs=node_attrs)
    G = nx.to_undirected(G)

    if task == 'graph':
        #print("arbitrary element:", nx.utils.arbitrary_element(G))
        #print("connected nodes:", sum(1 for node in nx.node_connected_component(G, nx.utils.arbitrary_element(G))))
        if nx.is_connected(G) is False:
            raise RuntimeError('[Input graph] is disconnected.')
        if radius is not None:
            G_subgraph = nx.ego_graph(G, node_id, radius=radius, undirected=False)
            G = generate_node_labels(G_subgraph)
        else:
            G = generate_node_labels(G)
    else:  # task == 'node'
        if radius is not None:
            G_subgraph = nx.ego_graph(G, node_id, radius=radius, undirected=False)
            extracted_id = [i for i in G_subgraph.nodes.keys()]
            G_subgraph = nx.relabel_nodes(G_subgraph, dict(zip(extracted_id, list(range(len(G_subgraph))))), copy=True)
            G = generate_node_labels(G_subgraph)
        else:
            extracted_id = [i for i in G.nodes.keys()]
            G = generate_node_labels(G)
        # index of the classification node in the extracted graph, for computing leaf_mask
        classification_node_id = extracted_id.index(node_id)

    is_clique_graph = True if len(list(G.edges)) == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2 else False

    # Create junction tree hierarchy
    G.graph = {'original': True}
    #zero_feature = [0.0] * data.num_node_features
    
    # TODO:  
    attr_feature_dict = {}
    for node_attr in node_attrs:
        attr_size = data[node_attr].size()
        if len(attr_size) == 1:
            num_attr_features = 1
            attr_feature = 0.0
        elif len(attr_size) == 2:
            num_attr_features = attr_size[1]
            if num_attr_features == 1:
                attr_feature = 0.0
            else:
                attr_feature = [0.0] * num_attr_features
        else:
            raise RuntimeError("Unsupported node attribute")
        
        attr_feature_dict[node_attr] = attr_feature
    
    #print("data.num_node_features:", data.num_node_features)
    
    #G_jth, root_nodes = generate_jth(G, zero_feature=attr_feature_dict)
    
    
    need_root_tree = True
    remove_edges_every_layer = True
    G_sampled, G_jth, root_nodes = sample_and_generate_jth(G, k=treewidth, zero_feature=attr_feature_dict,
                                                               copy_node_attributes=node_attrs,
                                                               need_root_tree=need_root_tree,
                                                               remove_edges_every_layer=remove_edges_every_layer,
                                                               verbose=False)
    '''
    # Convert back to torch Data (change first clique_has value to avoid TypeError when calling pyg_utils.from_networkx
    if is_clique_graph:  # clique graph
        G_jth.nodes[0]['clique_has'] = 0
    else:
        G_jth.nodes[0]['clique_has'] = [0]
    '''

    data_jth = from_networkx(G_jth)

    try:
        data_jth['diameter'] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth['diameter'] = 0
        print('junction tree hierarchy disconnected.')
        return data_jth

    if task == 'node':
        data_jth['classification_node'] = classification_node_id

    return data_jth, G_jth, root_nodes


def convert_room_object_graph_to_jth(data: Data, node_id=None, radius=None) -> Data:
    """
    Convert a room-object graph or its subgraph given id and radius (an ego graph) to junction tree hierarchy with
     leaf_mask attribute in addition to input feature x and label y. data_jth.leaf_mask is a BoolTensor of dimension
     [data_jth.num_nodes] specifying leaf nodes.
    :param data: torch_geometric.data.Data, input graph
    :param node_id: int, node id in the input graph to be classified (only used when task='node')
    :param radius: int, radius of the ego graph around classification node to be converted (only used when task='node')
    :returns: data_jth
    """
    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, 'node', node_id, radius)

    # Save leaf_mask
    leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
    for v, attr in G_jth.nodes('type'):
        if attr == 'node' and G_jth.nodes[v]['clique_has'] == data_jth['classification_node']:
            leaf_mask[v] = True
    data_jth['leaf_mask'] = leaf_mask
    data_jth.y = data.y[node_id]
    data_jth.y_room = data.room_mask[node_id]
    data_jth.y_object = data.object_mask[node_id]
    assert data_jth.y_room.item() != data_jth.y_object.item()

    data_jth.clique_has = None
    data_jth.type = None
    data_jth.classification_node = None
    return data_jth

def batch_edge_index(edge_index_init, n_nodes):
    """
    edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
    """
    n_examples = len(edge_index_init)
    edge_index = [edge_index_init[_i_] + sum(n_nodes[:_i_]) for _i_ in range(n_examples)]
    edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
    return edge_index

def convert_knowledge_graph_to_jths(data, min_diameter=None, max_diameter=None, treewidth = 1):
    

    #print("len(data):", len(data))
    qid, label, graph = data
    nc = len(graph)
    
    jths = []
    for node_input_ids, node_attention_mask, node_token_type_ids, node_output_mask, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type in graph:
        '''
        print("qid:", qid)
        print("label:", label.cpu().numpy())
        print("input_ids:", node_input_ids.size())
        print("attention_mask:", node_attention_mask.size())
        print("token_type_ids:", node_token_type_ids.size())
        print("output_mask:", node_output_mask.size())
        print("node_type_ids:", node_type_ids.size())
        print("node_scores:", node_scores.size())
        print("adj_lengths:", adj_lengths.cpu().numpy())
        print("special_nodes_mask:", special_nodes_mask.size())
        print("edge_index:", edge_index.size())
        print("edge_type:", edge_type.size())
        '''
        adj_lengths = adj_lengths.cpu().numpy()
        
        node_context_mask = torch.zeros(node_input_ids.size(0), dtype=torch.bool)
        node_context_mask[0] = True
    
        data = Data(x = node_input_ids[:adj_lengths,:], edge_index = edge_index, 
             node_type = node_type_ids[:adj_lengths], edge_type = edge_type, node_attention_mask = node_attention_mask[:adj_lengths,:], node_token_type_ids = node_token_type_ids[:adj_lengths,:], node_output_mask = node_output_mask[:adj_lengths,:],  node_scores = node_scores[:adj_lengths,:], node_context_mask=node_context_mask[:adj_lengths])

        print("data:", data)

        data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, 'graph', 0, None, node_attrs=["x", "node_type", "node_attention_mask", "node_token_type_ids", "node_output_mask", "node_scores", "node_context_mask"], edge_attrs=["edge_type"], treewidth=treewidth)
    
        
        # return empty lists if diameter is beyond specified bound
        try:
            data_jth['diameter'] = nx.diameter(G_jth)
        except nx.NetworkXError:
            data_jth['diameter'] = 0
            print('junction tree hierarchy disconnected.')
            return data_jth
        if (min_diameter is not None and data_jth.diameter < min_diameter) or \
                (max_diameter is not None and data_jth.diameter > max_diameter):
            return []
        
        data_jth.clique_has = None
        data_jth.type = None
        
        leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
        for node_id in range(data.num_nodes):
            for v, attr in G_jth.nodes('type'):
                if attr == 'node' and G_jth.nodes[v]['clique_has'] == node_id:
                    leaf_mask[v] = True
        data_jth['leaf_mask'] = leaf_mask
        
        #data_jth['y'] = data.y
        #data_jth['qid'] = data.qid
        
        print("data_jth:", data_jth)
        
        #print("data.x:" + str(list(data.x)) + ",data.edge_index:" + str(list(data.edge_index)) + ",data_jth.x:" + str(list(data_jth.x)) + ",data_jth.edge_index:" + str(list(data_jth.edge_index)))
        
        jths.append(data_jth)
    
    item_dict = {}
    for jth in jths:
        for key, value in jth:
            if key not in item_dict:
                item_dict[key] = []
            item_dict[key].append(value)
    n_nodes = [x.size(0) for x in item_dict["x"]]
    nc_mask = torch.cat([torch.tensor([n]*n_nodes[n]) for n in range(nc)])
    for key, value in item_dict.items():
        if key == "diameter":
            item_dict[key] = sum(value)//len(value)
        elif key == "edge_index":
            item_dict[key] = batch_edge_index(value, n_nodes)
        else:
            item_dict[key] = torch.cat(value)
    merged_data_jth = Data(qid=qid, y=label, nc_mask = nc_mask, **item_dict)
    
    print("merged_data_jth:", merged_data_jth)
    return merged_data_jth

def convert_room_object_graph_to_same_jths(data: Data, min_diameter=None, max_diameter=None):
    """
    Convert a room-object graph to the same junction tree hierarchy for all the original nodes.
    This function outputs three lists of jth's corresponding to original nodes in train_mask, val_mask and tes_mask.
    :param data: torch_geometric.data.Data, input graph
    :param min_diameter: int, minimum diameter of the results jth, below which this function returns three empty lists
    :param max_diameter: int, maximum diameter of the results jth, below which this function returns three empty lists
    :returns: train_list, val_list, test_list
    """
    assert isinstance(data, Data)
    train_list = []
    val_list = []
    test_list = []

    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, 'node', 0, None)

    # return empty lists if diameter is beyond specified bound
    try:
        data_jth['diameter'] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth['diameter'] = 0
        print('junction tree hierarchy disconnected.')
        return data_jth
    if (min_diameter is not None and data_jth.diameter < min_diameter) or \
            (max_diameter is not None and data_jth.diameter > max_diameter):
        return train_list, val_list, test_list

    # prepare modified copies of data_jth based on train/val/test masks
    data_jth.clique_has = None
    data_jth.type = None
    data_jth.classification_node = None
    for node_id in range(data.num_nodes):
        if data.train_mask[node_id].item() or data.val_mask[node_id].item() or data.test_mask[node_id].item():
            # create a copy of data_jth and related attributes
            data_jth_i = deepcopy(data_jth)
            leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
            for v, attr in G_jth.nodes('type'):
                if attr == 'node' and G_jth.nodes[v]['clique_has'] == node_id:
                    leaf_mask[v] = True
            data_jth_i['leaf_mask'] = leaf_mask
            data_jth_i.y = data.y[node_id]
            data_jth_i.y_room = data.room_mask[node_id]
            data_jth_i.y_object = data.object_mask[node_id]
            assert data_jth_i.y_room.item() != data_jth_i.y_object.item()
            # save to lists
            if data.train_mask[node_id].item() is True:
                train_list.append(data_jth_i)
            if data.val_mask[node_id].item() is True:
                val_list.append(data_jth_i)
            if data.test_mask[node_id].item() is True:
                test_list.append(data_jth_i)
    return train_list, val_list, test_list


def convert_dataset_to_junction_tree_hierarchy(dataset, task, min_diameter=None, max_diameter=None, radius=None, treewidth=1):
    """
    Convert a torch.dataset object or a list of torch.Data to a junction tree hierarchies.
    :param dataset:     a iterable collection of torch.Data objects
    :param task:            str, 'graph' or 'node'
    :param min_diameter:    int
    :param max_diameter:    int
    :param radius:          int, maximum radius of extracted sub-graphs for node classification
    :return: if task = 'graph', return a list of torch.Data objects in the same order as in dataset;
     else (task = 'node'), return a list of three such lists, for nodes and the corresponding subgraph in train_mask,
     val_mask, and test_mask respectively.
    """
    if task == 'graph':

        cpu_count = os.cpu_count()
        if radius is None:  # for nodes in the same graph, use the same junction tree hierarchy
            rtn_list = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count+4) as executor:
                for sub_dataset in dataset:
                    print("dataset size:", len(sub_dataset))
                    graph_list = list(executor.map(lambda data: convert_knowledge_graph_to_jths(data, min_diameter, max_diameter, treewidth), list(sub_dataset)))
                    rtn_list.append(graph_list)   
            '''
            # Sequential alternative for debug
            for sub_dataset in dataset:
                graph_list = []
                for data in list(sub_dataset):
                    graph = convert_knowledge_graph_to_jths(data, min_diameter, max_diameter)
                    graph_list.append(graph)
                rtn_list.append(graph_list)
            '''
            return rtn_list
        else:
            raise RuntimeError("Not implemented")
    elif task == 'node':
        train_list = []
        val_list = []
        test_list = []
        for data in dataset:
            print(data)
            if radius is None:  # for nodes in the same graph, use the same junction tree hierarchy
                train_graphs, val_graphs, test_graphs \
                    = convert_room_object_graph_to_same_jths(data, min_diameter, max_diameter)
                train_list += train_graphs
                val_list += val_graphs
                test_list += test_graphs
            else:               # otherwise, create jth for each node separately
                for i in range(data.num_nodes):
                    if data.train_mask[i].item() or data.val_mask[i].item() or data.test_mask[i].item():
                        data_jth = convert_room_object_graph_to_jth(data, node_id=i, radius=radius)
                        if (min_diameter is None or data_jth.diameter >= min_diameter) and \
                                (max_diameter is None or data_jth.diameter <= max_diameter):
                            if data.train_mask[i].item() is True:
                                train_list.append(data_jth)
                            elif data.val_mask[i].item() is True:
                                val_list.append(data_jth)
                            elif data.test_mask[i].item() is True:
                                test_list.append(data_jth)
        return [train_list, val_list, test_list]
    else:
        raise Exception("must specify if task is 'graph' or 'node' classification")


def get_subtrees_from_htree(data, G_htree, radius):
    """
    Segment sub-trees from input H-tree such that each sub-tree corresponds to a label node in the original graph.
    Note: if the original graph is disconnected, input H-tree should be computed from one of the connected component of
    the original graph.
    :param data: torch_geometric.data.Data, input graph
    :param G_htree: nx.Graph, H-tree decomposition of the (subsampled) original graph or one of the connected
    (subsampled) component of the original graph
    :param radius: int, furthest neighbor node from leaf nodes corresponding to a label node
    :return: train_list, val_list, test_list
    """
    # save leaf node indices for each node in the original graph
    leaf_nodes_list = [None] * data.num_nodes
    original_idx_set = set()      # indices of original nodes in data that are in G_jth
    for i, attr in G_htree.nodes('type'):
        if attr == 'node':
            original_idx = G_htree.nodes[i]['clique_has']
            original_idx_set.add(original_idx)
            if leaf_nodes_list[original_idx] is None:
                leaf_nodes_list[original_idx] = [i]
            else:
                leaf_nodes_list[original_idx].append(i)
    original_idx_list = list(original_idx_set)
    num_original_nodes = len(original_idx_list)

    # loop through each node in the original graph
    train_list = []
    val_list = []
    test_list = []
    data_mask = data.train_mask + data.val_mask + data.test_mask  # classification nodes
    progress_threshold = 0
    max_num_nodes = 0
    for j in range(num_original_nodes):
        original_idx = original_idx_list[j]
        if 100.0 * (j + 1) / num_original_nodes > progress_threshold + 10:
            progress_threshold += 10
        if data_mask[original_idx].item() is True:
            # segment subtree from the complete jth using specified radius from leaf nodes
            leaf_nodes = leaf_nodes_list[original_idx]
            G_subtree = nx.ego_graph(G_htree, leaf_nodes[0], radius=radius, undirected=False)
            for leaf_node in leaf_nodes[1:]:  # add other subtrees if there are multiple leaf nodes
                if G_subtree.number_of_nodes() == G_htree.number_of_nodes():
                    break
                H_subtree = nx.ego_graph(G_htree, leaf_node, radius=radius, undirected=False)
                G_subtree = nx.compose(G_subtree, H_subtree)
            extracted_id = [i for i in G_subtree.nodes.keys()]
            G_subtree = nx.relabel_nodes(G_subtree, dict(zip(extracted_id, list(range(len(G_subtree))))), copy=True)

            # convert G_subtree to torch data
            leaf_mask = torch.zeros(G_subtree.number_of_nodes(), dtype=torch.bool)
            for v, attr in G_subtree.nodes('type'):
                if attr == 'node' and G_subtree.nodes[v]['clique_has'] == original_idx:
                    leaf_mask[v] = True
                del G_subtree.nodes[v]['clique_has']
                del G_subtree.nodes[v]['type']
            data_subtree = pyg_utils.from_networkx(G_subtree)
            data_subtree.leaf_mask = leaf_mask
            data_subtree.y = data.y[original_idx]
            if nx.is_connected(G_subtree):
                data_subtree.diameter = nx.diameter(G_subtree)
            else:
                data_subtree.diameter = 0
            if data_subtree.num_nodes > max_num_nodes:
                max_num_nodes = data_subtree.num_nodes

            # save subtree
            if data.train_mask[original_idx].item() is True:
                train_list.append(data_subtree)
            elif data.val_mask[original_idx].item() is True:
                val_list.append(data_subtree)
            else:
                test_list.append(data_subtree)

    return train_list, val_list, test_list
