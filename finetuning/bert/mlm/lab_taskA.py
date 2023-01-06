# Importing Project Dependenices

## Pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn.functional as F

## Transformers
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import GPT2Tokenizer, OPTForSequenceClassification, AdamW, get_linear_schedule_with_warmup

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
wandb.login()


# Dataset Class
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Custom Loss
def loss_fn(outputs, targets):
    return torch.nn.functional.cross_entropy(outputs, targets)
    #return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# BERT Class
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('/home/kverma/spinning-storage/kverma/sem_eval_fin/output_dir/bert/lab/mlm/')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids, targets):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
       
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        
        return output 

# Helper Functions
## Loss Functions
def loss_fnc_taskB(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def loss_fnc(outputs, targets):
    return torch.nn.functional.cross_entropy(outputs, targets)

## Change Label
def change_taskA_label(string):

    if string == 'sexist':
        return 1
    else:
        return 0

def change_taskB_label(string):
    if '1.' in string:
        return [1, 0, 0, 0]
    elif '2.' in string:
        return [0, 1, 0, 0]
    elif '3.' in string:
        return [0, 0, 1, 0]
    elif '4.' in string:
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0]

# DataLoader, Optimizer & scheduler Function

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
        num_warmup_steps = round(total_steps*0.1), # Default value in run_glue.py == 0; changing to 10% of dataloader length
        num_training_steps = total_steps
    )

    return scheduler

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return elapsed_rounded

def train(train_dataloader, device, model, optimizer, scheduler):

    t0 = time.time()

    total_train_loss = 0
    
    for _, data in enumerate(train_dataloader):
        
        input_ids = data['ids'].to(device, dtype = torch.long)
        
        input_mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].type(torch.LongTensor)
        targets = targets.to(device)

        outputs = model(input_ids, input_mask, token_type_ids, targets)

        loss = loss_fn(outputs, targets)
        
        loss.backward()

        total_train_loss += loss.item()

        wandb.log({'train_batch_loss':loss.item()})

        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    wandb.log({'avg_train_loss':loss.item()})

    training_time = format_time(time.time() - t0)

    return avg_train_loss, training_time

def validate(model, val_dataloader, device):

    t0 = time.time()

    model.eval()

    #Tracking Variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    fin_targets=[]
    fin_outputs=[]

    with torch.no_grad():
        for _,data in enumerate(val_dataloader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].type(torch.LongTensor)
            targets = targets.to(device)
            outputs = model(ids, mask, token_type_ids, targets)
            loss = loss_fn(outputs, targets)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        
            total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_dataloader)

    wandb.log({'avg_val_loss':avg_val_loss})

    return fin_outputs, fin_targets



def train_validate(model, device, epochs, train_dataloader, val_dataloader, optimizer, scheduler):

    wandb.watch(model)
    model.to(device)
    
    model.train()

    for epoch_i in tqdm(range(0, epochs)):

        print("")
        print("Running Train....")

        avg_train_loss, training_time = train(train_dataloader, device, model, optimizer, scheduler)

        print("")
        print("Running Validation....")

        outputs, targets = validate(model, val_dataloader, device)
        
        outputs = np.argmax(outputs, axis=1).flatten()

        accuracy = metrics.accuracy_score(targets, outputs)
        
        wandb.log({'val_accuracy':accuracy})

        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        
        wandb.log({'f1_macro':f1_score_macro})

        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
    
    print("*"*5,"\tTraining Complete")

def main_train_val():
    model = BERTClass()

    default_config = {
            'batch_size':16,
            'learning_rate':1e-5,
            'epochs':2,
            'weight_decay':0.01
            }

    wandb.init(project="bert-lab-mlm", entity="dcu-semval10", config=default_config)

    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    weight_decay = wandb.config.weight_decay

    MAX_LEN = 256

    
    # Get Data
    dataset_path = '/home/kverma/spinning-storage/kverma/sem_eval/datasets/'
    data = pd.read_csv(dataset_path+'train_all_tasks.csv')

    taskA = data[['rewire_id', 'text', 'label_sexist']]
    taskA['label'] = taskA['label_sexist'].apply(change_taskA_label)

    tokenizer = BertTokenizer.from_pretrained('/home/kverma/spinning-storage/kverma/sem_eval_fin/output_dir/bert/lab/mlm/')

    train_size = 0.8
    train_dataset = taskA.sample(frac=train_size,random_state=200)
    test_dataset=taskA.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(taskA.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    train_dataloader = DataLoader(training_set, **train_params)
    val_dataloader = DataLoader(testing_set, **test_params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer = ret_optim(learning_rate, model, weight_decay)
    scheduler = ret_scheduler(train_dataloader, optimizer, epochs)

    wandb.init(project='bert-lab-mlm', entity='dcu-semval10', config=default_config)

    seed_val = 42
    np.random.seed(seed_val)

    train_validate(model, device, epochs, train_dataloader, val_dataloader, optimizer, scheduler)



if __name__ == '__main__':

    sweep_config = {
            'method' : 'bayes',
            'name': 'sweep',
            'metric': {
                'name': 'f1_macro',
                'goal': 'maximize',
            },
            'parameters':{
                'learning_rate': {
                    'values': [5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 1e-3, 3e-3, 5e-3]
                },
                'batch_size': {
                    'values': [8, 16, 32]
                },
                'epochs': {
                    'values': [2,3,4,5]
                },
                'weight_decay' : {
                    'values': [0.01, 0.0001, 0.0005, 0.001]
                }
            },
            'early_terminate':{
                'type':'hyperband',
                'min_iter':2
                }
        }

    
    sweep_id = wandb.sweep(sweep = sweep_config, project='bert-lab-mlm')

    wandb.agent(sweep_id = sweep_id, function = main_train_val)

