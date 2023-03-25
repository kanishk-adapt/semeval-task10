# Importing Project Dependencies
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import wandb
import argparse
wandb.login()
import torch
import random
import torch
from collections import Counter

# Helper Functions
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def get_sentence_pairs(sentences, labels, pairs):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
	# labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sentences)):      
        currentSentence = sentences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList==label)[0][0]])
        posSentence = sentences[idxB]
		  # prepare a positive pair and update the sentences and labels
		  # lists, respectively
        pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSentence = sentences[np.random.choice(negIdx)]
		  # prepare a negative pair of images and update our lists
        pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))
  
    return (pairs)

def change_label_taskc(string):

    if '1.1' in string:
        return 0
    elif '1.2' in string:
        return 1
    elif '2.1' in string:
        return 2
    elif '2.2' in string:
        return 3
    elif '2.3' in string:
        return 4
    elif '3.1' in string:
        return 5
    elif '3.2' in string:
        return 6
    elif '3.3' in string:
        return 7
    elif '3.4' in string:
        return 8
    elif '4.1' in string:
        return 9
    else:
        return 10

def label_submission(pred):
    if pred == 0:
        return  '1.1 threats of harm'
    elif pred == 1:
        return '1.2 incitement and encouragement of harm'
    elif pred == 2:
        return '2.1 descriptive attacks'
    elif pred == 3:
        return '2.2 aggressive and emotive attacks'
    elif pred == 4:
        return '2.3 dehumanising attacks & overt sexual objectification'
    elif pred == 5:
        return '3.1 casual use of gendered slurs, profanities, and insults'
    elif pred == 6:
        return '3.2 immutable gender differences and gender stereotypes'
    elif pred == 7:
        return '3.3 backhanded gendered compliments'
    elif pred == 8:
        return '3.4 condescending explanations or unwelcome advice'
    elif pred == 9:
        return '4.1 supporting mistreatment of individual women'
    else:
        return '4.2 supporting systemic discrimination against women as a group'


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune MLM for taskC")
    
    parser.add_argument(
        '--train_path',
        type=str,
        default='/home/kverma/spinning-storage/kverma/sem_eval/dev_task_a_entries.csv',
        help='enter the path of training file'
    )

    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='enter the path of validation file'
    )

    parser.add_argument(
        '--task_test_path',
        type=str,
        default=None,
        help='enter the path of validation file'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='lr',
        help='enter the path of validation file'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='bert-base-uncased',
        help='enter the path of validation file'
    )

    parser.add_argument(
            '--save_name',
            type=str,
            default='hbert_lr_1',
            )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    train_path = args.train_path
    test_path = args.test_path
    task_test_path = args.task_test_path
    algorithm = args.algorithm
    model_dir = args.model_dir
    save_name = args.save_name

    # Hyperparam as per wandb run https://wandb.ai/dcu-semval10/hb-u-sfit/runs/xnor72ko?workspace=user-kank
    num_itr = 10
    epochs = 2
    batch_size = 32
    warmup_steps = 10
    weight_decay = 0.1
    learning_rate = 5e-5

    print('CSV to dataframe\n')
    train_df = pd.read_csv(train_path, on_bad_lines="skip")
    test_df = pd.read_csv(test_path)

    taskc = pd.read_csv(task_test_path)

    train_df = train_df[train_df['label_vector']!='none']

    print('Labels converted\n')
    train_df['label'] = train_df['label_vector'].apply(change_label_taskc)
    test_df['label_new'] = test_df['label'].apply(change_label_taskc)

    x_train = train_df['text'].values.tolist()
    y_train = train_df['label'].values.tolist()

    x_test = test_df['text'].values.tolist()
    y_test = test_df['label_new'].values.tolist()

    x_taskc = taskc['text'].values.tolist()
    print('Generating Sentence Pairs\n')
    train_examples = []
    for x in range(num_itr):
        train_examples = get_sentence_pairs(np.array(train_df['text'].values.tolist()), np.array(train_df['label'].values.tolist()), train_examples)

    print('Converting to Dataloader\n')
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    model = SentenceTransformer(model_dir)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, scheduler = 'WarmupLinear', warmup_steps = warmup_steps, optimizer_class = torch.optim.AdamW, optimizer_params = {'lr': learning_rate}, weight_decay = weight_decay, show_progress_bar=True)
    X_train = model.encode(x_train)
    X_test = model.encode(x_test)
    X_taskc = model.encode(x_taskc)

    if 'lr' in algorithm:
        sgd =  LogisticRegression()
        print('Now training Logistic Regression\n')
        sgd.fit(X_train, y_train)
        y_test_pred = sgd.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average="macro")
        y_taskc_pred = sgd.predict(X_taskc)
        print('Testing on Dev-set F1-macro:\t{}'.format(test_f1))
    else:
        knn = KNeighborsClassifier(n_neighbors=5,)
        print('Now training K Neighbors Classifier\n')
        knn.fit(X_train, y_train)
        y_test_pred = knn.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average="macro")
        print('Testing on Dev-set F1-macro:\t{}'.format(test_f1))
        y_taskc_pred = knn.predict(X_taskc)

    
    label_pred = [label_submission(pred) for pred in y_taskc_pred]
    taskc['label_pred'] = label_pred
    
    submission = taskc[['rewire_id', 'label_pred']]
    submission.to_csv(save_name+'.csv', index=False)



