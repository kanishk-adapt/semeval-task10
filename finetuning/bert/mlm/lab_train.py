from helpers import * 

def train():
    path = '/home/kverma/spinning-storage/kverma/sem_eval/datasets/'
    train_df, val_df = load_file(path)

    X_train, y_train = get_Xy(train_df)
    X_val, y_val = get_Xy(val_df)
    model_dir = '/home/kverma/spinning-storage/kverma/sem_eval_fin/output_dir/bert/lab/mlm/'

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

    train_inputs, train_masks = preprocess(tokenizer, X_train)
    val_inputs, val_masks = preprocess(tokenizer, X_val)

    # Convert labels to Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    

    default_config = {
            'batch_size': 16,
            'learning_rate': 1e-5,
            'epochs': 2
            }

    wandb.init(project="bert-lab-mlm", entity="dcu-semval10", config=default_config)

    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ret_model(model_dir)
    wandb.watch(model)
    model.to(device)

    train_dataloader, val_dataloader = ret_dataloader(batch_size, train_dataset, val_dataset)

    optimizer = ret_optim(learning_rate, model)
    scheduler = ret_scheduler(train_dataloader, optimizer, epochs)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        print("")
        print('========= Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,} of {:>5,}. Elapsed: {:}'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids = None,
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

        training_time = format_time(time.time() - t0)

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

        for batch in val_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():

                outputs = model(b_input_ids,
                                token_type_ids = None,
                                attention_mask = b_input_mask, 
                                labels = b_labels)

                loss, logits = outputs['loss'], outputs['logits']

            total_eval_loss += loss.item()


            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        
        avg_val_loss = total_eval_loss / len(val_dataloader)


        val_time = format_time(time.time() - t0)
        wandb.log({'val_accuracy': avg_val_accuracy, 'avg_val_loss': avg_val_loss})

        print(" Validation Loss: {0:.2f}".format(avg_val_loss))
        print(" Validation tookL {:}".format(val_time))


        training_stats.append({
                                'epoch': epoch_i + 1,
                                'Training Loss': avg_train_loss,
                                'Valid. Loss': avg_val_loss,
                                'Valid. Accur.': avg_val_accuracy,
                                'Training time': training_time,
                                'Validation Time': val_time
                                })
    print("")
    print("Training Complete")

    #return model, training_stats



if __name__ == '__main__':

    sweep_config = {
            'method' : 'grid',
            'name': 'sweep',
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize',
            },
            'parameters':{
                'learning_rate': {
                    'values': [5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
                },
                'batch_size': {
                    'values': [8, 16, 32]
                },
                'epochs': {
                    'values': [2,3,4,5]
                }
            }
        }

    sweep_id = wandb.sweep(sweep = sweep_config, project='bert-lab-mlm')

    wandb.agent(sweep_id = sweep_id, function = train)
    #model, training_stats = train(model_dir, 'ru-mbert', train_dataset, val_dataset, test_dataset)

