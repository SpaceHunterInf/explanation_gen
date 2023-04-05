import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import json
from functools import partial

class ExpDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label = args['label']
    
    def __getitem__(self, index):
        if self.label == 'entailment':
            prompt = 'Explain why the premise entails the hypothesis. '
        elif self.label == 'contradiction':
            prompt = 'Explain why the premise contradicts the hypothesis. '
        elif self.label == 'neutral':
            prompt = 'Explain why the premise is neutral with respect to the hypothesis. '

        x = {}

        if 't5' in self.args['model_name']:
            x['input_text'] = prompt + 'Premise: ' + self.data[index]['premise'] + ' ' + 'Hypothesis: ' + self.data[index]['hypothesis']
            x['output_text'] = 'Explanation: ' + self.data[index]['explanation']
            x['premise'] = self.data[index]['premise']
            x['hypothesis'] = self.data[index]['hypothesis']
            if self.args['auto_eval'] == True:
                x['highlighted_premise'] = self.data[index]['highlighted_premise']
                x['highlighted_hypothesis'] = self.data[index]['highlighted_hypothesis']

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
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["label_ids"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):
    if args['data'] == 'eSNLI':
        path_train = 'data/eSNLI/train_e-snli-{}.json'.format(args['label'])
        path_dev = 'data/eSNLI/dev_e-snli-{}.json'.format(args['label'])
        path_test = 'data/eSNLI/test_e-snli-{}.json'.format(args['label'])
    elif args['data'] == 'IBM':
        path_train = 'data/IBMDebate/filtered/train_ibm-nli-{}.json'.format(args['label'])
        path_dev = 'data/IBMDebate/filtered/dev_ibm-nli-{}.json'.format(args['label'])
        path_test = 'data/IBMDebate/filtered/test_ibm-nli-{}.json'.format(args['label'])

    with open(path_train, 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(path_dev, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    with open(path_test, 'r', encoding='utf-8') as f:
        test = json.load(f)

    train_dataset = ExpDataset(args, train, tokenizer)
    dev_dataset = ExpDataset(args, dev, tokenizer)
    test_dataset = ExpDataset(args, test, tokenizer)

    # if "gpt" in args['model_name']:
    #     train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    #     test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    #     dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    if 't5' in args['model_name']:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)


    return train_loader, dev_loader, test_loader