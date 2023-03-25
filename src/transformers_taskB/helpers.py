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


def change_label_1(string):

    if '1' in string:
        return 1
    else:
        return 0

def change_label_2(string):

    if '2' in string:
        return 1
    else:
        return 0

def change_label_3(string):

    if '3' in string:
        return 1
    else:
        return 0

def change_label_4(string):

    if '4' in string:
        return 1
    else:
        return 0


def load_file(train_path, val_path, string):

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    train = train[train['label_category']!='none']
    val = val[val['label_category']!='none']

    if '1' in string:

        train['label'] = train['label_category'].apply(change_label_1)
        val['label'] = val['label_category'].apply(change_label_1)

    elif '2' in string:

        train['label'] = train['label_category'].apply(change_label_2)
        val['label'] = val['label_category'].apply(change_label_2)

    elif '3' in string:

        train['label'] = train['label_category'].apply(change_label_3)
        val['label'] = val['label_category'].apply(change_label_3)

    else:

        train['label'] = train['label_category'].apply(change_label_4)
        val['label'] = val['label_category'].apply(change_label_4)

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

def training(model, project_name, sweep_defaults, train_dataset, val_dataset):

    batch_size = sweep_defaults['batch_size']
    learning_rate = sweep_defaults['learning_rate']
    epochs = sweep_defaults['epochs']
    weight_decay = sweep_defaults['weight_decay']

    wandb.init(project=project_name, entity='dcu-semval10', config=sweep_defaults)
    
    training_stats = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:\t', device)

    model.to(device)
    wandb.watch(model)

    train_dataloader, val_dataloader = ret_dataloader(batch_size, train_dataset, val_dataset)

    print('Torch data loaded...\n')

    optimizer = ret_optim(learning_rate, model, weight_decay)
    
    print('Optimizer Loaded...\n')

    scheduler = ret_scheduler(train_dataloader, optimizer, epochs)

    print('Scheduled Loaded...\n')

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    total_t0 = time.time()

    print('Training Begins:\t')

    for epoch_i in tqdm(range(0, epochs), desc='Training'):

        print("")
        print('========= Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask = b_input_mask,
                            labels = b_labels)

            loss, logits = outputs['loss'], outputs['logits']

            wandb.log({'train_batch_loss':loss.item()})

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()


        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = time.time() - t0

        wandb.log({'avg_train_loss': avg_train_loss})


        ##################################
        #          VALIDATION            #
        ##################################

        print("")
        print("Running Validation....")

        t0 = time.time()

        model.eval()

        #Tracking Variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        
        fin_targets = []
        fin_outputs = []

        for batch in tqdm(val_dataloader, desc='Validation'):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():

                outputs = model(b_input_ids,
                                attention_mask = b_input_mask,
                                labels = b_labels)

                loss, logits = outputs['loss'], outputs['logits']

            total_eval_loss += loss.item()


            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
            output_logit = np.argmax(logits, axis=1).flatten()
            fin_targets.extend(label_ids)
            fin_outputs.extend(output_logit)

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

        avg_val_f1 = metrics.f1_score(fin_targets, fin_outputs, average="macro")
        
        print("\tF1-macro: {:0.4f}".format(avg_val_f1))

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))


        avg_val_loss = total_eval_loss / len(val_dataloader)

        val_time = time.time() - t0 
        wandb.log({'val_accuracy': avg_val_accuracy, 'avg_val_loss': avg_val_loss, 'avg_f1_macro': avg_val_f1})

        print(" Validation Loss: {0:.2f}".format(avg_val_loss))
        print(" Validation tookL {:}".format(val_time))


        training_stats.append({
                                'epoch': epoch_i + 1,
                                'Training Loss': avg_train_loss,
                                'Valid. Loss': avg_val_loss,
                                'Valid. Accur.': avg_val_accuracy,
                                'Valid. F1 Macro': avg_val_f1,
                                'Training time': training_time,
                                'Validation Time': val_time
                                })
    print("")
    print("Training Complete")


    return model, training_stats
