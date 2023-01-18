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

# Helper Functions

def change_label_taskA(label):

    if label == 'sexist':
        return 1
    else:
        return 0

def load_file(train_path, val_path):

    train = pd.read_csv(train_path)

    train['label'] = train['label_sexist'].apply(change_label_taskA)

    if 'k8020' in val_path:

        val = pd.read_csv(val_path)

        val['label'] = val['label_sexist'].apply(change_label_taskA)

    else:

        val = pd.read_csv(val_path)

        val['label_sexist'] = val['label'].apply(change_label_taskA)

        val = val.rename(columns={'label_sexist': 'label', 'label':'label_sexist'})

    train = train[['rewire_id', 'text', 'label']]
    val = val[['rewire_id', 'text', 'label']]

    return train, val

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

def ret_dataloader(batch_size, train_dataset, val_dataset):

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    val_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    return train_dataloader, val_dataloader

def ret_optim(learning_rate, model, weight_decay):

    optimizer = AdamW( 
                    model.parameters(),
                    lr = learning_rate,
                    eps = 1e-8,
                    weight_decay = weight_decay,
                    )

    return optimizer

def ret_scheduler(train_dataloader, optimizer, epochs):
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0, # Default value in run_glue.py
        num_training_steps = total_steps
    )

    return scheduler

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return elapsed_rounded  #str(datetime.timedelta(seconds=elapsed_rounded))
