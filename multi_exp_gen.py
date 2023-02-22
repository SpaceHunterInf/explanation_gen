import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from copy import deepcopy
import json
from tqdm import tqdm
from config import *
from multi_exp_data_loader import *
import random, os, json, re

def generate_output(args, tokenizer, model, test_loader, save_path, label):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save = []
    model.to('cuda')
    for batch in tqdm(test_loader):
        with torch.no_grad():
            if 't5' in args['model_name']:
                #print(batch)
                model.cuda()
                outputs = model.generate(input_ids=batch["input_ids"].to(device='cuda'),
                                attention_mask=batch["attention_mask"].to(device='cuda'),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=50,
                                )
                outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for idx in range(len(outputs_text)):
                    save.append(outputs_text[idx])
    
    with open(os.path.join(save_path,'results_{}.json'.format(label)), 'w') as f:
        f.write(json.dumps(save, indent=2))
        f.close()

if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    labels = ['entailment', 'contradiction', 'neutral']

    for l in labels:
        model_path = 'save/t5-smallt5-small0.0001_epoch_5_seed_557_{}'.format(l)
        if "t5" in args["model_name"]:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))

        dataset = prepare_data(args, 'data/chaosNLI/chaosNLI_snli.json', tokenizer, l)
        save_path = 'data/chaosNLI/SNLI'
        print("test start...")
        #evaluate model
        generate_output(args, tokenizer, model, dataset, save_path, l)