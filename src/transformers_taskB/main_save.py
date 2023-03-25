from helpers import *
import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune MLM for taskA")

    parser.add_argument(
        '--train_path',
        type=str,
        default='/home/kverma/spinning-storage/kverma/sem_eval/dev_task_a_entries.csv',
        help='enter the path of training file'
    )

    parser.add_argument(
        '--validation_path',
        type=str,
        default=None,
        help='enter the path of validation file'
    )

    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='enter the path of validation file'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='bert-base-uncased',
        help='enter the path of model'
    )

    parser.add_argument(
            '--model_name',
            type=str,
            default='bert',
            help='enter the name of the model'
            )

    parser.add_argument(
            '--label_number',
            type=str,
            default='1',
            help="Please enter the label number for Task-B"
            )

    parser.add_argument(
            '--wandb_project',
            type=str,
            default='bert',
            help="Please enter the wandb-project name"
            )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=16,
            help="enter the batch-size"
            )
    
    parser.add_argument(
            '--epochs',
            type=int,
            default=1,
            help="enter the number of epochs"
            )

    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
            help="enter the learning rate"
            )

    parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.001,
            help="enter the weight decay"
            )

    parser.add_argument(
            '--save_dir',
            type=str,
            default=None,
            help="Please enter the directory to save the trained model"
            )

    args = parser.parse_args()

    return args

def ret_test_dataloader(batch_size, test_dataset):

    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    return test_dataloader

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

def get_predictions(tokenizer, model, dataframe, batch_size):

    preds = []
    truth_list = []
    for i in tqdm(range(len(dataframe)), desc="Testing"):

        text = dataframe['text'].iloc[i]
        label = dataframe['new_label'].iloc[i]
        truth_list.append(label)
        text = text.lower()

        test_inputs, test_masks = preprocess(tokenizer, [text])
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_dataloader = ret_test_dataloader(batch_size, test_dataset)

        probs = get_preds(model, test_dataloader)

        y_preds = np.argmax(probs, axis=1)

        preds.extend(y_preds)

    test_f1 = metrics.f1_score(truth_list, preds)

    text = 'F1 score on test-set is: {}'.format(test_f1)

    return preds, truth_list, text

    

def main():

    train, val = load_file(train_path, val_path, label_number)
 
    X_train = list(train['text'])
    y_train = list(train['label'])

    X_val = list(val['text'])
    y_val = list(val['label'])

    if 'opt' in model_name.lower():

        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, do_lower_case=True)
        
        model = ret_opt_model(model_dir)

    else:

        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

        model = ret_bert_model(model_dir)

    train_inputs, train_masks = preprocess(tokenizer, X_train)
    val_inputs, val_masks = preprocess(tokenizer, X_val)

    # Convert labels to Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    model = ret_bert_model(model_dir)

    model, training_stats = training(model, project_name, sweep_defaults, train_dataset, val_dataset)

    return model, tokenizer, training_stats
    

if __name__ == "__main__":

    args = parse_args()

    train_path = args.train_path
    val_path = args.validation_path
    test_path = args.test_path

    label_number = args.label_number
    model_dir = args.model_dir
    project_name = args.wandb_project
    model_name = args.model_name
    
    learning_rate = args.learning_rate
    epochs = args.epochs
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    save_dir = args.save_dir

    sweep_defaults = {
            'learning_rate': learning_rate,
            'batch_size' : batch_size,
            'epochs': epochs,
            'weight_decay': weight_decay
        }
    
    model, tokenizer, training_stats = main()

    print("\t Now Testing")
    
    test_df = pd.read_csv(test_path)

    if '1' in label_number:
        test_df['new_label'] = test_df['label'].apply(change_label_1)
    elif '2' in label_number:
        test_df['new_label'] = test_df['label'].apply(change_label_2)
    elif '3' in label_number:
        test_df['new_label'] = test_df['label'].apply(change_label_3)
    else:
        test_df['new_label'] = test_df['label'].apply(change_label_4)
    
    test_df = test_df[['text', 'new_label']]

    preds, truth_list, text = get_predictions(tokenizer, model, test_df, batch_size)
    
    print("\n\n")
    print(text)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(save_dir+'training_stats.txt', 'w') as files:
        json.dump(training_stats, files)

    
    print('\nModel & training-stats saved at: {}'.format(save_dir))
    
