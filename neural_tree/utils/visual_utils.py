'''
Created on Jun. 27, 2022

@author: yfang
'''
import networkx as nx
import matplotlib.pyplot as plt
import torch
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, MobileBertTokenizer, AlbertTokenizer)
from neural_tree.utils.data_utils import MODEL_NAME_TO_CLASS
def visualize_graph(model_name, node_hidden_states, embeddings, embedding_projection, node_context_mask, leaf_mask, edge_index, save_file):
    model_type = MODEL_NAME_TO_CLASS[model_name]
    
    tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer, 'mobilebert': MobileBertTokenizer}.get(model_type)

    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    num_subplots = len(node_hidden_states)
    for subplot_id, node_hidden_state in enumerate(node_hidden_states):
        G = nx.Graph()
        
        for node_id, node_seq_embs in enumerate(node_hidden_state):
            
            sentence = []
            for word_emb in node_seq_embs:
                word_emb = torch.tensor(word_emb).unsqueeze(0).to(embeddings.weight.data.device)
                if embedding_projection is not None:
                    #print("word_emb.size():", word_emb.size())
                    #print("embedding_projection.bias.data.size():", embedding_projection.bias.data.size())
                    #print("embedding_projection.weight.data.size():", embedding_projection.weight.data.size())
                    #print("torch.linalg.pinv(embedding_projection.weight.data).size():", torch.linalg.pinv(embedding_projection.weight.data).size())
                    
                    word_emb = torch.matmul(word_emb - embedding_projection.bias.data, torch.t(torch.linalg.pinv(embedding_projection.weight.data)))
                distance = torch.norm(embeddings.weight.data - word_emb, dim=1)
                word_id = torch.argmin(distance)
                sentence.append(word_id)
            sentence = tokenizer.decode(sentence)
            
            
            G.add_node(node_id, color='lightblue', label=sentence)
    
        
        G.add_edges_from(zip(edge_index[0],edge_index[1]))
        
        #print(G.nodes(data=True))
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        plt.subplot(num_subplots, 1, subplot_id+1)
        nx.draw(G, node_color=colors, with_labels=True, font_color='black', labels={node[0]:node[1]['label'] for node in G.nodes(data=True)})

    plt.savefig(save_file)

