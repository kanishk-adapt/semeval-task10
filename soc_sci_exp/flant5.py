from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json
import pandas as pd

with open('fin_sample.json', 'r') as pos_file:
    data = json.load(pos_file)


model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')


def get_flan_response(prompt):

    input_prompt = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**input_prompt)

    return outputs

def change(string):

    if string == True:

        return 'sexist'
    else:
        return 'not sexist'


final_data = []

for i in tqdm(range(len(data))):
    

    text = data[i]['text']
    rewire_id = data[i]['rewire_id']
    gpt_pred = data[i]['is_sexist']
    gpt_pred = change(gpt_pred)
    gpt_words = data[i]['words_or_signals']
    gpt_ration = data[i]['rationale']

    # Binary Prompt 
    
    prompt_1 = "Answer the following question, Identify if the following sentence is sexist or not sexist, Sentence: {}".format(text)
    outputs_1 = get_flan_response(prompt_1)
    flan_resp = tokenizer.batch_decode(outputs_1, skip_special_tokens=True)
    flan_1 = ' '.join(flan_resp)
    
    # Words or phrases

    prompt_2 = "Answer the following question, If the following sentence {} , is {}, identify key phrases or words or signals from the sentence.".format(text, flan_1)
    outputs_2 = get_flan_response(prompt_2)
    flan_resp = tokenizer.batch_decode(outputs_2, skip_special_tokens=True)
    flan_2 = ', '.join(flan_resp)

    # Decision making

    #prompt_3 = "Answer the following question, Write a statement why is the following sentence sexist or not sexist, sentence: {}".format(text)
    
    prompt_3 = "Answer the following question, Describe why the following sentence is {}. Sentence: {}".format(flan_1, text)
    outputs_3 = get_flan_response(prompt_3)
    flan_resp = tokenizer.batch_decode(outputs_3, skip_special_tokens=True)
    flan_3 = ' '.join(flan_resp)


    resp_2_3 = "I base my decision of identifying this sentences as {}, because of the words or phrases in the sentence '{}'. Also the sentence implies {}".format(flan_1, flan_2, flan_3.lower())

    

    tup = (rewire_id, text, gpt_pred, gpt_words, gpt_ration, flan_1, flan_2, resp_2_3)

    final_data.append(tup)



final = pd.DataFrame(final_data, columns=['rewire_id', 'text', 'gpt_prediction', 'gpt_words', 'gpt_rationale','flant5_prediction', 'flant5_words', 'flant5_rationale'])

final.to_csv('both_system.csv', index=False)



