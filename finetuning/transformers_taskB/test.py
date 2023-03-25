#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) 2019, 2022, 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Kanishk Verma

# Importing Project Dependencies
## Transformer - Hugging Face
import transformers
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from transformers import GPT2Tokenizer, OPTForSequenceClassification, AdamW, get_linear_schedule_with_warmup

## Pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn.functional as F

## Metrics
from sklearn import metrics

## Utilities
import numpy as np
import pandas as pd
import random
import time
import os
import datetime
import json
import wandb
from datetime import datetime
from tqdm import tqdm
import wandb
import argparse
wandb.login()

from collections import Counter

def get_preds(model, test_dataloader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)
    batch_count = 0
    all_logits = []
    #print('For each batch in our test set...')
    # For each batch in our test set...
    for batch in test_dataloader:
        batch_count += 1
        
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits[0])

    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def preprocess(tokenizer, data):

    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens = True,
            max_length = 128,
            return_tensors = 'pt',
            return_attention_mask = True,
            pad_to_max_length = True,
            truncation = True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])


    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def ret_test_dataloader(batch_size, test_dataset):

    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    return test_dataloader

def ret_bert_model(model_dir):

    model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels = 2
            )
    
    return model

def ret_opt_model(model_dir):

    model = OPTForSequenceClassification.from_pretrained( 
                                                        model_dir,
                                                        num_labels=2
                                                        )
    return model

def change_label_taskb(string):

    if '1' in string:
        return 0
    elif '2' in string:
        return 1
    elif '3' in string:
        return 2
    else:
        return 3

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune MLM for taskA")

    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='enter the path of validation file'
    )

    parser.add_argument(
            '--model_name',
            type=str,
            default='bert',
            help='enter the name of the model'
            )
    
    parser.add_argument(
        '--model_dir_lab1',
        type=str,
        default='bert-base-uncased',
        help='enter the path of model'
    )

    parser.add_argument(
        '--model_dir_lab2',
        type=str,
        default='bert-base-uncased',
        help='enter the path of model'
    )

    parser.add_argument(
        '--model_dir_lab3',
        type=str,
        default='bert-base-uncased',
        help='enter the path of model'
    )

    parser.add_argument(
        '--model_dir_lab4',
        type=str,
        default='bert-base-uncased',
        help='enter the path of model'
    )

    args = parser.parse_args()

    return args


def change_pred_list(lists):

    if lists == [1,0,0,0]:

        return '1. threats, plans to harm and incitement'

    elif lists == [0,1,0,0]:

        return '2. derogation'

    elif lists == [0,0,1,0]:

        return '3. animosity'

    else:

        return '4. prejudiced discussions'


if __name__ == "__main__":

    args = parse_args()
    test_path = args.test_path
    model_name = args.model_name
    model_dir_lab1 = args.model_dir_lab1
    model_dir_lab2 = args.model_dir_lab2
    model_dir_lab3 = args.model_dir_lab3
    model_dir_lab4 = args.model_dir_lab4
    batch_size = 8
    
    test_df = pd.read_csv(test_path)

    if 'opt' in model_name.lower():

        tokenizer = GPT2Tokenizer.from_pretrained(model_dir_lab1, do_lower_case=True)
        
        model_1 = ret_opt_model(model_dir_lab1)
        model_2 = ret_opt_model(model_dir_lab2)
        model_3 = ret_opt_model(model_dir_lab3)
        model_4 = ret_opt_model(model_dir_lab4)

        fname = 'opt_res'

    else:

        tokenizer = BertTokenizer.from_pretrained(model_dir_lab1, do_lower_case=True)

        model_1 = ret_bert_model(model_dir_lab1)
        model_2 = ret_bert_model(model_dir_lab2)
        model_3 = ret_bert_model(model_dir_lab3)
        model_4 = ret_bert_model(model_dir_lab4)

        fname = 'hbert_res'

    pred_list = []
    truth_list = []
    data = []
    for i in tqdm(range(len(test_df)), desc="Testing"):

        text = test_df['text'].iloc[i]
        r_id = test_df['rewire_id'].iloc[i]

        test_inputs, test_masks = preprocess(tokenizer, [text])
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_dataloader = ret_test_dataloader(batch_size, test_dataset)

        probs_1 = get_preds(model_1, test_dataloader)
        y_pred_1 = np.argmax(probs_1, axis=1)
        probs_2 = get_preds(model_2, test_dataloader)
        y_pred_2 = np.argmax(probs_2, axis=1)
        probs_3 = get_preds(model_3, test_dataloader)
        y_pred_3 = np.argmax(probs_3, axis=1)
        probs_4 = get_preds(model_4, test_dataloader)
        y_pred_4 = np.argmax(probs_4, axis=1)
        
        y_preds = list((y_pred_1[0], y_pred_2[0], y_pred_3[0], y_pred_4[0]))

        if y_preds.count(1) > 1:

            high_prob_index = np.argmax([probs_1[0], probs_2[0], probs_3[0], probs_4[0]])

            if high_prob_index == 0:

                y_preds = [1,0,0,0]

            elif high_prob_index == 1:

                y_preds = [0,1,0,0]

            elif high_prob_index == 2:

                y_preds = [0,0,1,0]

            else:

                y_preds = [0,0,0,1]
        

        y_pred = change_pred_list(y_preds)
        
        tup = (r_id, y_pred)
        data.append(tup)


    data_df = pd.DataFrame(data, columns=['rewire_id', 'label_pred'])
    data_df.to_csv('ensemble_taskb.csv',index=False)
