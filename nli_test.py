from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification, pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import json, os
import random
from tqdm import tqdm
import torch.nn.functional as F

def get_probs(model, tokenizer, premise, hypothesis):
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                    max_length=1024,
                                                    return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    
    return torch.softmax(outputs[0], dim=1)[0]

if __name__ == '__main__':

    # hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    processed_data = []

    proper_data = []
    improper_data = []
    for d in processed_data:
        if d['fluency'] == True and d['consistency'] == False:
            if d['explanation_with_premise'] == d['generation_label']:
                proper_data.append(d)
            else:
                improper_data.append(d)

    with open('flan-t5/processed.json', 'r') as f:
        flant5 = json.load(f)

    with open('flan-t5-prompted/processed.json', 'r') as f:
        flant5prompted = json.load(f)

    with open('bloom/processed.json', 'r') as f:
        bloom = json.load(f)

    with open('gpt-3.5/processed.json', 'r') as f:
        gpt = json.load(f)

    processed_data += flant5 + flant5prompted + bloom + gpt

    shifted = 0
    changed = 0
    counter = 0
    prev_probs = []
    after_probs = []
    target_probs = []

    proper_data = []
    improper_data = []
    for d in processed_data:
        if d['fluency'] == True and d['consistency'] == False:
            if d['explanation_with_premise'] == d['generation_label']:
                proper_data.append(d)
            else:
                improper_data.append(d)

    labels = ['entailment', 'neutral', 'contradiction']
    for i in tqdm(proper_data, desc='proper_data'):
        if not 'ibm' in i['uid']:
            probs = get_probs(model, tokenizer, i['premise'], i['hypothesis'])
            prev_probs.append(torch.log(probs))
            e_probs = get_probs(model, tokenizer, i['premise'] + i['explanation'], i['hypothesis'])
            after_probs.append(torch.log(e_probs))

            target_prob = [i['label_counter'][x] for x in ['e', 'n', 'c']]
            target_probs.append(torch.tensor(target_prob)/100)

            prev_label = labels[torch.argmax(probs).item()]
            e_label = labels[torch.argmax(e_probs).item()]

            if prev_label != i['generation_label']:
                counter += 1
                if e_label == i['generation_label']:
                    changed += 1
                
                label_idx = labels.index(i['generation_label'])

                if e_probs.tolist()[label_idx] > probs.tolist()[label_idx]:
                    shifted += 1
            if proper_data.index(i) == 177:
                break

    with open(hg_model_hub_name.split('/')[-1]+'human_eval.txt','w') as f:
        f.write(str(changed/counter), str(shifted/counter), str(counter))
        f.close()