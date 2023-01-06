# Importing Project Dependenices

## Transformer - Hugging Face
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

## Metrics - Sklearn
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, matthews_corrcoef

## Plots - Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

## Utilities
from collections import Counter
import wandb
import numpy as np
import pandas as pd
import random
import time
import os
import datetime
import regex as re
import json

'''

with open('/home/kverma/spinning-storage/kverma/sem_eval/training/contractions.json') as file_1:
    x = json.load(file_1)

with open('/home/kverma/spinning-storage/kverma/sem_eval/training/ncontractions.json') as file_2:
    y = json.load(file_2)

contractions = {**x, **y}
'''

# HELPER FUNCTIONS
def load_file(path):
    df = pd.read_csv(path+'train_all_tasks.csv')
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    val = df[~msk]
    
    return train, val

def change_label(label):

    if label == 'sexist':
        return 1
    else:
        return 0

def clean(text):

    #Remove Contractions
    tokens = text.split()

    for i in range(len(tokens)):
        for k,v in contractions.items():
            if k in tokens[i]:
                tokens[i] = v

    ntext = ' '.join(tokens)

    #Remove [url], [user]
    ntext = re.sub("[[\w]+]", ' ', ntext)
    
    #Space between punctuations
    ntext = re.sub("([.,!?()'])", r' \1 ', ntext)

    ntokens = ntext.split()
    newText = ' '.join(ntokens)

    return newText

def get_Xy(df):
    # rewire_id,text,label_sexist,label_category,label_vector
    #df['clean_text'] = df['text'].apply(clean)

    X = df['text'].values

    df['label_taskA'] = df['label_sexist'].apply(change_label)

    y = df['label_taskA'].values

    return X,y

def preprocess(tokenizer, data):

    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens = True,
            max_length = 256,
            return_tensors = 'pt',
            return_token_type_ids = False,
            return_attention_mask = True,
            pad_to_max_length = True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    
    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def ret_model(model_dir):

    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels = 2,
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


def ret_optim(learning_rate, model):
    optimizer = AdamW(
        model.parameters(),
        lr = learning_rate,
        eps = 1e-8
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
    return str(datetime.timedelta(seconds=elapsed_rounded))


def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    print('Perform a forward pass on the trained BERT model to predict probabilities on the test set.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    #print('Put the model into the evaluation mode. The dropout layers are disabled during the test time.')

    model.eval()
    batch_count = 0
    all_logits = []
    #print('For each batch in our test set...')
    # For each batch in our test set...
    for batch in test_dataloader:
        batch_count += 1
        #print('Batch: ', batch_count)
        # Load batch to GPU
        #print('Load batch to GPU')
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        #print('Compute logits')
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits[0])
    
    #print('Concatenate logits from each batch')
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    #print('Apply softmax to calculate probabilities')
    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def save_prediction(model, test_dataloader, X_test, y_test, output_dir, filename):
    print('Predicting')
    probs = bert_predict(model, test_dataloader)
    preds = np.where(probs[:, 1] >= 0.5, 1, 0)
    print('Storing results: ', output_dir)
    
    df = pd.DataFrame(columns=['text', 'truth', 'pred'])
    df['text'] = X_test.text.values
    df['truth'] = y_test.label.values
    df['pred'] = preds

    df.to_csv(output_dir+filename+'_scores.csv', index=False)


