import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import json
from copy import deepcopy
from functools import partial

class MultiExplainDataset(Dataset):
    def __init__(self,args, data, tokenizer, l, source='ChaoSNLI'):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
        self.label = l
        self.source = source
    
    def __getitem__(self, index):
        if self.label == 'entailment':
            prompt = 'Explain why the premise entails the hypothesis. '
        elif self.label == 'contradiction':
            prompt = 'Explain why the premise contradicts the hypothesis. '
        elif self.label == 'neutral':
            prompt = 'Explain why the premise is neutral with respect to the hypothesis. '
        x = deepcopy(self.data[index])

        if 't5' in self.args['model_name']:
            if self.source == 'ChaoSNLI':
                x['input_text'] = prompt + 'Premise: ' + self.data[index]['example']['premise'] + ' ' + 'Hypothesis: ' + self.data[index]['example']['hypothesis']
            elif self.source == 'IBM':
                x['input_text'] = prompt + 'Premise: ' + self.data[index]['premise'] + ' ' + 'Hypothesis: ' + self.data[index]['hypothesis']
        return x

    def __len__(self):
        return len(self.data)

def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["input_ids"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]

    return batch_data

def prepare_data(args, file_name, tokenizer, l, source='ChaoSNLI'):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = MultiExplainDataset(args, data, tokenizer, l, source)
    if 't5' in args['model_name']:
        dataset = DataLoader(dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    
    return dataset