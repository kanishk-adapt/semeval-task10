from helpers import *
from training import *
from test import *

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
            help = "enter the path of the test file"
            )

    parser.add_argument(
            '--model_name',
            type=str,
            default='bert',
            help='enter the name of the model'
            )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='bert-base-uncased',
        help='enter the path of model'
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
            '--wandb_project',
            type=str,
            default='bert',
            help="Please enter the wandb-project name"
            )

    parser.add_argument(
            '--save_dir',
            type=str,
            default=None,
            help="Please enter the directory to save the trained model"
            )

    args = parser.parse_args()

    return args

def change_label_to_string(label_list):
    
    new_list = []

    for label in label_list:
        if label == 1:

            new_list.append('sexist')

        else:

            new_list.append('not sexist')
    
    return new_list

if __name__ == "__main__":

    args = parse_args()

    train_path = args.train_path
    val_path = args.validation_path
    test_path = args.test_path
    model_name = args.model_name
    model_dir = args.model_dir

    learning_rate = args.learning_rate
    epochs = args.epochs
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    
    save_dir = args.save_dir

    if val_path == None:

        train_val = pd.read_csv(train_path)
        train_val['label'] = train_val['label_sexist'].apply(change_label_taskA)

        train = train_val.loc[:round(len(train_val)*0.9)]
        val = train_val.loc[round(len(train_val)*0.9):]

    else:

        train, val = load_file(train_path, val_path)

    #train = train.iloc[:10]
    #val = val.iloc[:2]
    
    X_train = train.text.values
    y_train = train.label.values

    X_val = val.text.values
    y_val = val.label.values
    
    if model_name == 'OPT' or model_name == 'opt':

        model = ret_opt_model(model_dir)

        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, do_lower_case=True)

    else:

        model = ret_bert_model(model_dir)

        tokenizer = BertTokenizer.from_pretrained(model_dir)

    
    train_inputs, train_masks = preprocess(tokenizer, X_train)
    val_inputs, val_masks = preprocess(tokenizer, X_val)

    # Convert labels to Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    

    sweep_defaults = {

            'learning_rate': learning_rate,
            'batch_size' : batch_size,
            'epochs': epochs,
            'weight_decay': weight_decay
        }
    
    print('\n\tBegin Training....\n')

    project_name = args.wandb_project
    
    model, training_stats = training(model, project_name, sweep_defaults, train_dataset, val_dataset)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(save_dir+'training_stats.txt', 'w') as files:
        json.dump(training_stats, files)

    print('\nModel & training-stats saved at: {}'.format(save_dir))
    
    if test_path != None:
        print('\n\tBegin Testing...\n')
        
        test_df = pd.read_csv(test_path)

        #test_df = test_df.loc[:2]
        
        test_preds, test_truth = get_predictions(tokenizer, model, test_df, batch_size)
        
        test_f1 = metrics.f1_score(test_preds,test_truth)


        print('\n\tF1-macro Score: {:0.4f}'.format(test_f1))

        test_preds_lab = change_label_to_string(test_preds)
        test_truth_lab = change_label_to_string(test_truth)

        test_df['preds'] = test_preds_lab
        test_df['truth'] = test_truth_lab


        test_df.to_csv(model_name+'_test_results.csv', index=False)

    else:

        print('\n\tNo Testing File provided...')


