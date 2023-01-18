from helpers import *

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

            if step % 40 == 0 and not step == 0:

                #elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,} of {:>5,}. Elapsed: {:}'.format(step, len(train_dataloader), elapsed))

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