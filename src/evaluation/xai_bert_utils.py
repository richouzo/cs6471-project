import gc
import argparse
import numpy as np
import datetime
import time

import spacy
import pandas as pd
from sklearn.metrics import f1_score

from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils.preprocess_utils import *
from src.training.train_utils import train_model, test_model
from src.evaluation.test_save_stats import *
from src.evaluation.xai_utils import VisualizationDataRecordCustom

from src.utils.utils import *

import captum
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from typing import Any, Iterable, List, Tuple, Union
from IPython.core.display import HTML, display


def construct_input_ref_pair_from_raw(text, tokenizer, device):
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used for the end of a sentence
    cls_token_id = tokenizer.cls_token_id # A token used for the start of a sentence

    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    ref_input_ids = torch.tensor(ref_input_ids, device=device).unsqueeze(0)

    return input_ids, ref_input_ids


def construct_input_ref_pair(text, tokenizer, device):
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used for the end of a sentence
    cls_token_id = tokenizer.cls_token_id # A token used for the start of a sentence

    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = text_ids
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * (len(text_ids) - 2) + [sep_token_id]
    ref_input_ids = torch.tensor(ref_input_ids, device=device).unsqueeze(0)

    return input_ids, ref_input_ids


def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    # https://github.com/pytorch/captum/issues/150#issuecomment-549022512
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.transformer(embedding_output, attention_mask, head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = sequence_output.mean(axis=1)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions) 


class BertModelWrapper(nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        out = self.model.relu(self.model.linear1(pooled_output))
        logits = self.model.linear2(out)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)


def interpret_sentence(model_wrapper, tokenizer, ig, sentence, label, original_idx, 
                       vis_data_records_ig, device, raw_text=False):
    torch.cuda.empty_cache()
    gc.collect()
    model_wrapper.eval()
    model_wrapper.zero_grad()

    # print('sentence: ', sentence)

    if raw_text:
        input_ids, ref_input_ids = construct_input_ref_pair_from_raw(sentence, tokenizer, device)
    else:
        input_ids, ref_input_ids = construct_input_ref_pair(sentence, tokenizer, device)
    input_embedding = model_wrapper.model.bert.embeddings(input_ids).to(device)
    

    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, return_convergence_delta=True)

    # print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))
    

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist()) 
    # print('tokens:', tokens)
    add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta, original_idx, vis_data_records_ig)
    
    torch.cuda.empty_cache()
    del attributions_ig, tokens, input_ids, input_embedding, pred, pred_ind
    gc.collect()

def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, original_idx, 
                                   vis_data_records, class_names=["Neutral","Hate"]):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()
    
    # storing couple samples in an array for visualization purposes visualization.VisualizationDataRecord
    datarecord = VisualizationDataRecordCustom(attributions,
                                                pred,
                                                class_names[pred_ind],
                                                class_names[label],
                                                class_names[1],
                                                attributions.sum(),       
                                                tokens[:len(attributions)],
                                                delta, 
                                                original_idx,)
    vis_data_records.append(datarecord)





### Attribution scores functions ###

def interpret_sentence_with_stats(model_wrapper, tokenizer, ig, sentence, label, original_idx, device, raw_text=False):
    torch.cuda.empty_cache()
    gc.collect()

    if raw_text:
        input_ids, ref_input_ids = construct_input_ref_pair_from_raw(sentence, tokenizer, device)
    else:
        input_ids, ref_input_ids = construct_input_ref_pair(sentence, tokenizer, device)
    input_embedding = model_wrapper.model.bert.embeddings(input_ids).to(device)

    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, return_convergence_delta=True)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist()) 

    torch.cuda.empty_cache()
    return (attributions_ig, delta, tokens, input_ids, input_embedding, pred, pred_ind, label, original_idx)

def model_explainability_bert_with_stats(model, tokenizer, ig, df, max_samples, device, from_notebook=True):
    """
    Computing words importance for each sample in df and save stats
    """
    if from_notebook:
        from tqdm.notebook import trange, tqdm
    else:
        import tqdm
    print("\n\n**MODEL EXPLAINABILITY**\n")
    print("Computing words importance for each sample... ")

    stats = []

    model.eval()
    model.zero_grad()

    for idx in tqdm(range(max_samples)):
        sentence = df.iloc[idx].text
        if len(sentence) > 512: continue # GPU memory saturated if the sentence is too long
        label = df.iloc[idx].true_label
        original_idx = df.iloc[idx].original_index
        with torch.set_grad_enabled(False):
            local_stats = interpret_sentence_with_stats(model, tokenizer, ig, sentence, label, original_idx, device)
            
            stats.append(local_stats)

    print("Computations completed.")
    return stats

def process_attributions_stats(attributions):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()
    return attributions
