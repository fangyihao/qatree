'''
Created on Jun. 27, 2022

@author: yfang
'''
import networkx as nx
import matplotlib.pyplot as plt
import torch
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, MobileBertTokenizer, AlbertTokenizer)
from util.data_util import MODEL_NAME_TO_CLASS
import numpy as np
import textwrap
import cv2
import os
import torch_geometric.utils as pyg_utils
from transformers import (
             LogitsProcessorList,
             MinLengthLogitsProcessor,
             BeamSearchScorer,
         )
from transformers.pytorch_utils import torch_int_div
from transformers.generation_stopping_criteria import StoppingCriteriaList
from torch import nn

CONTEXT_COLOR = (255/256,226/256,187/256)
LEAF_COLOR = (242/256,244/256,193/256)
OTHER_COLOR = (194/256,232/256,247/256)

# TODO: BeamSearchScorer fails
class BeamSearch:
    def __init__(self, device, bos_token_id, eos_token_id, pad_token_id, num_beams = 3, batch_size=1):
        self.device = device
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_beams = num_beams
        self.batch_size = batch_size
        
        self.sentence_ids = torch.ones((num_beams, 1), device=device, dtype=torch.long)
        self.sentence_ids = self.sentence_ids * bos_token_id
        
        self.beam_scorer = BeamSearchScorer(
             batch_size=batch_size,
             num_beams=num_beams,
             device=device,
        )
        self.logits_processor = LogitsProcessorList(
             [
                 MinLengthLogitsProcessor(5, eos_token_id=eos_token_id),
             ]
        )
        self.stopping_criteria = StoppingCriteriaList()

        self.beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        self.beam_scores[:, 1:] = -1e9
        self.beam_scores = self.beam_scores.view((batch_size * num_beams,))
        
    def __call__(self, next_token_scores):
        
        next_token_scores = torch.ones((self.num_beams*self.batch_size, 1),dtype=torch.float, device=self.device)*next_token_scores
        
        next_token_scores_processed = self.logits_processor(self.sentence_ids, next_token_scores)
        
        print("beam_scores:", self.beam_scores)
        #print("beam_scores:", self.beam_scores[:, None].expand_as(next_token_scores))
        #print("next_token_scores:", next_token_scores)
        
        next_token_scores = next_token_scores_processed + self.beam_scores[:, None].expand_as(next_token_scores)
        
        #print("next_token_scores 2:", next_token_scores)
        
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)
        
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
        )
        print("next_token_scores:", next_token_scores)
        print("next_tokens:", next_tokens)
        
        self.next_indices = torch_int_div(next_tokens, vocab_size)
        self.next_tokens = next_tokens % vocab_size
        beam_outputs = self.beam_scorer.process(
                        self.sentence_ids,
                        next_token_scores,
                        self.next_tokens,
                        self.next_indices,
                        pad_token_id=self.pad_token_id,
                        eos_token_id=self.eos_token_id,
                    )
        self.beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        
        print("beam_next_tokens:", beam_next_tokens)
        print("beam_idx:", beam_idx)
        
        self.sentence_ids = torch.cat([self.sentence_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    
    def finalize(self):
        sequence_outputs = self.beam_scorer.finalize(
            self.sentence_ids,
            self.beam_scores,
            self.next_tokens,
            self.next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.stopping_criteria.max_length,
        )
        return sequence_outputs["sequences"][0]
        
    def reset(self):
        self.sentence_ids = torch.ones((self.num_beams, 1), device=self.device, dtype=torch.long)
        self.sentence_ids = self.sentence_ids * self.bos_token_id
        
        self.beam_scores = torch.zeros((self.batch_size, self.num_beams), dtype=torch.float, device=self.device)
        self.beam_scores[:, 1:] = -1e9
        self.beam_scores = self.beam_scores.view((self.batch_size * self.num_beams,))

def animate_graph(config, 
                  node_hidden_states, 
                  embeddings, 
                  embedding_projection, 
                  node_context_mask, 
                  leaf_mask, 
                  batch_mask, 
                  edge_index, 
                  qid, 
                  save_file_path="animation", 
                  visualize_context = False, 
                  decoder="greedy_search",
                  keep_frame_image = True,
                  viz_label = "heat+l0_text"
                  ):
    
    def heat2str(heat):
        return ''.join(['\u25a7']*int(max(heat-50, 0)//10))
    
    if decoder == "beam_search":
        beam_search = BeamSearch(device = embeddings.weight.data.device, bos_token_id = config.bos_token_id, eos_token_id = config.eos_token_id, pad_token_id = config.pad_token_id)
    
    model_name = config._name_or_path
    model_type = MODEL_NAME_TO_CLASS[model_name]
    
    tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer, 'mobilebert': MobileBertTokenizer}.get(model_type)

    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    nl = len(node_hidden_states)
    
    pos_map = {}
    vid_map = {}
    
    emb_map = {}
    text_map = {}
    
    mbs = np.max(batch_mask)+1
    for layer_id, node_hidden_state in enumerate(node_hidden_states):
        node_ids = np.array(list(range(len(node_hidden_state))))
        
        for batch_id in range(mbs):     
            bt_node_hidden_state = node_hidden_state[(batch_mask == batch_id)]
            bt_node_ids = node_ids[(batch_mask == batch_id)]
            bt_node_context_mask = node_context_mask[(batch_mask == batch_id)]
            bt_leaf_mask = leaf_mask[(batch_mask == batch_id)]
            G = nx.Graph()
            
            for node_id, node_seq_embs, is_context, is_leaf  in zip(bt_node_ids, bt_node_hidden_state, bt_node_context_mask, bt_leaf_mask):
                
                if decoder == "beam_search":
                    beam_search.reset()
                elif decoder == "greedy_search":
                    sentence = []
                else:
                    raise NotImplementedError()
                
                emb_map[str(layer_id)+"_"+str(batch_id)+"_"+str(node_id)] = node_seq_embs
                
                for word_emb in node_seq_embs:
                    word_emb = torch.tensor(word_emb).unsqueeze(0).to(embeddings.weight.data.device)
                    if embedding_projection is not None:
                        #print("word_emb.size():", word_emb.size())
                        #print("embedding_projection.bias.data.size():", embedding_projection.bias.data.size())
                        #print("embedding_projection.weight.data.size():", embedding_projection.weight.data.size())
                        #print("torch.linalg.pinv(embedding_projection.weight.data).size():", torch.linalg.pinv(embedding_projection.weight.data).size())
                        word_emb = torch.matmul(word_emb - embedding_projection.bias.data, torch.t(torch.linalg.pinv(embedding_projection.weight.data)))
                    
                    distance = torch.norm(embeddings.weight.data - word_emb, dim=1)
                    if decoder == "beam_search":
                        next_token_scores = nn.functional.log_softmax(-distance, dim=-1)
                        #print("next_token_scores:", next_token_scores)
                        beam_search(next_token_scores)
                    elif decoder == "greedy_search":
                        word_id = torch.argmin(distance)
                        sentence.append(word_id)
                    else:
                        raise NotImplementedError()
                if decoder == "beam_search":
                    sentence = beam_search.finalize()
                #print("sentence:", sentence)
                sentence = ("[{}] ".format(str(is_context)) if visualize_context else "") + tokenizer.decode(sentence, skip_special_tokens=True)
                
                
                text_map[str(layer_id)+"_"+str(batch_id)+"_"+str(node_id)] = sentence
                
                if layer_id > 0:
                    heat = np.linalg.norm(node_seq_embs - emb_map[str(layer_id-1)+"_"+str(batch_id)+"_"+str(node_id)])
                else:
                    heat = 0
                
                if viz_label == "heat":
                    label = heat2str(heat)
                elif viz_label == "text":
                    label = textwrap.fill(sentence, 40, break_long_words=False)
                elif viz_label == "heat+l0_text":
                    text = text_map[str(0)+"_"+str(batch_id)+"_"+str(node_id)]
                    text = textwrap.fill(text, 40, break_long_words=False)
                    label = heat2str(heat) + '\n' + text
                else:
                    raise NotImplementedError()
                
                #label = textwrap.fill(label, 40, break_long_words=False)
                
                G.add_node(node_id, color=(CONTEXT_COLOR if is_context>=0 else LEAF_COLOR) if is_leaf else OTHER_COLOR, label=label)
                
                
        
            edges = []
            for u, v in zip(edge_index[0],edge_index[1]):
                if u in bt_node_ids or v in bt_node_ids:
                    edges.append((u,v))
            
            G.add_edges_from(edges)
            
            if layer_id == 0:
                pos_map[batch_id]=nx.spring_layout(G)
            plt.figure(figsize=(60,60))
            plt.title("layer {}".format(layer_id))
            colors = [node[1]['color'] for node in G.nodes(data=True)]
            #plt.subplot(num_subplots, 1, subplot_id+1)
            nx.draw(G, pos=pos_map[batch_id], node_size=10000, node_color=colors, with_labels=True, font_color='black', labels={node[0]:node[1]['label'] for node in G.nodes(data=True)})
            #plt.show()
            
            frame_file = "{}/{}_{}.jpg".format(save_file_path, qid[batch_id], layer_id)
            plt.savefig(frame_file)
                
            plt.close()
            
            frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
            if layer_id == 0:
                vid = cv2.VideoWriter("{}/{}.avi".format(save_file_path, qid[batch_id]), 
                     cv2.VideoWriter_fourcc(*'MJPG'),
                     1, frame.shape[:2])
                vid_map[batch_id ] = vid
            vid_map[batch_id].write(frame)
            if not keep_frame_image:
                if layer_id > 0:
                    os.system("rm {}".format(frame_file))
            if layer_id == nl - 1:
                vid_map[batch_id].release()
 
def draw_graph(model_name, data, graph_type, save_file_path="animation", visualize_context_mask=True):
    
    if graph_type == "kg":
        node_attrs=["x", "node_type", "node_attention_mask", "node_token_type_ids", "node_output_mask", "node_scores", "node_context_mask"]
    elif graph_type == "jth":
        node_attrs=["x", "node_type", "node_attention_mask", "node_token_type_ids", "node_output_mask", "node_scores", "node_context_mask", "leaf_mask"] 
    else:
        raise Exception("Not implemented!")
    qid = data.qid
    G = pyg_utils.to_networkx(data, node_attrs=node_attrs)
    G = nx.to_undirected(G)
    
    model_type = MODEL_NAME_TO_CLASS[model_name]
    
    tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer, 'mobilebert': MobileBertTokenizer}.get(model_type)

    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    
    for node in G.nodes(data=True):
        node_input_ids = node[1]["x"]
        sentence = ""
        if visualize_context_mask:
            sentence = "[{}] ".format(str(node[1]["node_context_mask"])) 
        sentence += tokenizer.decode(node_input_ids) 
        sentence = textwrap.fill(sentence, 40, break_long_words=False)
        
        node[1]["label"] = sentence
        
        if graph_type == "kg":
            if node[1]["node_context_mask"] >= 0:
                node[1]["color"] = CONTEXT_COLOR
            else:
                node[1]["color"] = LEAF_COLOR
        elif graph_type == "jth":
            
            #print("leaf_mask:", node[1]["leaf_mask"])
            #print("node_context_mask:", node[1]["node_context_mask"])
            
            if bool(node[1]["leaf_mask"]):
                if node[1]["node_context_mask"] >= 0:
                    node[1]["color"] = CONTEXT_COLOR
                else:
                    node[1]["color"] = LEAF_COLOR
            else:
                node[1]["color"] = OTHER_COLOR
        else:
            raise Exception("Not implemented!")
            
        
        
    pos=nx.spring_layout(G)
    plt.figure(figsize=(60,60))
    
    colors = [node[1]['color'] for node in G.nodes(data=True)]
    #plt.subplot(num_subplots, 1, subplot_id+1)
    nx.draw(G, pos=pos, node_size=10000, node_color=colors, with_labels=True, font_color='black', labels={node[0]:node[1]['label'] for node in G.nodes(data=True)})
    #plt.show()
    
    frame_file = "{}/{}_{}.jpg".format(save_file_path, qid, graph_type)
    plt.savefig(frame_file)
        
    plt.close()
    
    
