from helpers import *

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

def change_label_taskA(label):

    if label == 'sexist':

        return 1

    else:

        return 0

def get_predictions(tokenizer, model, dataframe, batch_size):

    preds = []

    dataframe['label_sexist'] = dataframe['label'].apply(change_label_taskA)
    
    truth_list = []

    for i in tqdm(range(len(dataframe)), desc="Testing"):

        text = dataframe['text'].iloc[i]
        rewire_id = dataframe['rewire_id'].iloc[i]
        true = dataframe['label_sexist'].iloc[i]
        truth_list.append(true)
        text = text.lower()

        test_inputs, test_masks = preprocess(tokenizer, [text])
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_dataloader = ret_test_dataloader(batch_size, test_dataset)

        probs = get_preds(model, test_dataloader)

        y_preds = np.argmax(probs, axis=1)

        preds.extend(y_preds)


    return preds, truth_list


def get_f1_macro(preds, truth):
    

    print(metrics.f1_score(preds,truth))





