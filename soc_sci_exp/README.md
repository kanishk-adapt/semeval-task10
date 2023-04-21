# Explanations

1. flant5.py -> to generate prediction, important words and rationale from 'google/flan-t5-large' 
2. lig.py -> to gather layer integrated gradient important scores for HateBERT model
3. fin_sample.json -> json file containing rewire-id, chat-gpt responses as {"is_sexist": binary, "words_or_signals": [list of words], "rationale": string of explanation"}
4. both_system.csv -> responses from both FlanT5 and ChatGPT
5. model_imp.csv -> layer-wise integrated important scores for each token in text for HateBERT model generated by running lig.py
6. EDOS-System-Annotator-1/2/3.xlsx -> annotations by 3 social science experts