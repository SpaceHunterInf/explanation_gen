import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import json
from functools import partial

class ChaosNLI(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):

        x = {}
        x['premise'] = self.data[index]['example']['premise']
        x['hypothesis'] = self.data[index]['example']['hypothesis']
        x['e'] = self.data[index]['example']['entailment_explanation']
        x['n'] = self.data[index]['example']['neutral_explanation']
        x['c'] = self.data[index]['example']['contradiction_explanation']
        x['input_text'] = 'Premise: ' + x['premise'] + ' ' + 'Hypothesis: ' + x['hypothesis'] + 'Entaiment ' + x['e'] + 'Neutral ' + x['n'] + 'Contradiction' + x['c']
        
        x['label'] = torch.tensor(self.data[index]['label_dist'])
        x['input_ids'], x['attention_mask'] = roberta_preprocess(self.tokenizer, x, 514, self.args['original'])
        return x

    def __len__(self):
        return len(self.data)
    

def roberta_preprocess(tokenizer, input_dict, length, original=False):
    if original:
        tokenized_input_seq_pair = tokenizer.encode_plus(input_dict['premise'], input_dict['hypothesis'],
                                                    max_length=length,
                                                    return_token_type_ids=True, truncation=True)

        input_ids = tokenized_input_seq_pair['input_ids']
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = tokenized_input_seq_pair['token_type_ids']
        attention_mask = tokenized_input_seq_pair['attention_mask']
    else:
        inputs = tokenizer(input_dict['input_text'])
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        
    padding_length = length - len(input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
        
    input_ids = torch.tensor(input_ids)#.unsqueeze(0)
    attention_mask = torch.tensor(attention_mask)#.unsqueeze(0)
    return input_ids, attention_mask
    
def prepare_data(args, tokenizer):
    with open('flan-t5-prompted/chaosNLI_snli_augmented.json','r') as f:
        data = json.load(f)

    train_dataset = ChaosNLI(args, data[:int(0.8*len(data))], tokenizer)
    dev_dataset = ChaosNLI(args, data[int(0.8*len(data)):int(0.9*len(data))], tokenizer)
    test_dataset = ChaosNLI(args, data[int(0.9*len(data)):], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, num_workers=16)

    return train_loader, dev_loader, test_loader