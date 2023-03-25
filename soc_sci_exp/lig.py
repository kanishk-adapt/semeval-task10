# Importing Project Dependencies
import pandas as pd 
from transformers import BertTokenizer, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
from collections import Counter
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
torch.cuda.empty_cache()

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

import spacy 
nlp = spacy.load("en_core_web_sm")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper Functions

def remove_unwanted_spaces(sent):

    sentTokens = sent.split()

    if '(created' in sentTokens:

        sentTokens.remove('(created')

    elif 'created' in sentTokens:

        sentTokens.remove('created')

    elif 'https' in sentTokens:

        sentTokens.remove('https')
    
    else:

        sentTokens = sentTokens

    return ' '.join(sentTokens)

def remove_blanks(df):
    
    return df[df['texts'].str.len() > 1]

def further_clean(df):
    
    df_v1 = df.dropna()

    df_v1['texts'] = df_v1['texts'].str.lower()

    df_v1['texts'] = df_v1['texts'].apply(remove_unwanted_spaces)
    df_v2 = remove_blanks(df_v1)
    df_v3 = df_v2.dropna()

    return df_v3

def get_bert_model(output_dir):

    model = BertForSequenceClassification.from_pretrained(output_dir, output_hidden_states = True, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    return model, tokenizer



def get_special_tokens(tokenizer):
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    
    return ref_token_id, sep_token_id, cls_token_id


def construct_input_ref_pair(text, tokenizer, ref_token_id, sep_token_id, cls_token_id):

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids), tokens

def construct_whole_bert_embeddings(model, input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
    
    return input_embeddings, ref_input_embeddings

def custom_forward_func2(input_emb, attention_mask=None, position=0):
    pred = model(inputs_embeds=input_emb, attention_mask=attention_mask, )
    pred = pred[position]
    return pred.max(1).values


def get_gradient_maps(data, tokenizer, model):

    ref_token_id, sep_token_id, cls_token_id = get_special_tokens(tokenizer)

    grad_imp_scores = []

    for i in range(len(data)):

        #row_idx = i
        rewire_id = data['rewire_id'].iloc[i]
        sent = data['text'].iloc[i]
        #sent = data['texts'].iloc[i]

        input_ids, ref_input_ids, sep_id, tokens = construct_input_ref_pair(sent, tokenizer, ref_token_id, sep_token_id, cls_token_id)

        input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(model, input_ids, ref_input_ids)


        for layer in range(model.config.num_hidden_layers):

            lc = LayerConductance(custom_forward_func2, model.bert.encoder.layer[layer])

            layer_attributions = lc.attribute(input_embeddings, ref_input_embeddings)

            for head in range(0,12):

                for tkn_i in range(len(tokens)):

                    mean_gradient_importance_score = torch.mean(torch.norm(layer_attributions[1][0][head][tkn_i])).cpu().detach().numpy()

                    tup = (rewire_id, tokens[tkn_i], layer, head, mean_gradient_importance_score)

                    grad_imp_scores.append(tup)
            
            torch.cuda.empty_cache()

    grad_df = pd.DataFrame(grad_imp_scores, columns=['rewire_id', 'token', 'layer', 'head','mean_gradient_important_score'])

    return grad_df


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('both_system.csv')

data = data[['rewire_id', 'text']]

bert_dir = 'enter directory'
model, tokenizer = get_bert_model(bert_dir)

model.to(device)

temp = get_gradient_maps(data, tokenizer, model)

print(temp.head(2))

temp.to_csv('model_imp.csv', index=False)

print('Done')
