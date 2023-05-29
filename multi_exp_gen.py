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
from prompt_tuning import Soft_Prompt_T5
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
                                max_length=128,
                                )
                outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif 'soft-prompt' in args['model_name']:
                model.cuda()
                outputs = outputs = model.generate(batch["input_ids"].to(device='cuda'),
                                                   batch["attention_mask"].to(device='cuda'), 
                                                   tokenizer.eos_token_id, 
                                                   128)
                outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif 'bloom' in args['model_name']:
                model.cuda()
                outputs = model.generate(input_ids=batch["input_ids"].to(device='cuda'),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=128
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
    labels = ['entailment', 'contradiction']

    for l in labels:
        model_path = 'flan-t5/ibm_{}'.format(l)
        if "t5" in args["model_name"]:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "bloom" in args["model_name"]:
            model = BloomForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "soft-prompt" in args["model_name"]:
            t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'.format(l))
            tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base', bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            t5.resize_token_embeddings(new_num_tokens=len(tokenizer))
            model = Soft_Prompt_T5(t5, tokenizer)
            model.load_state_dict(torch.load('flan-t5-prompted/IBMflan-t50.0001_epoch_10_seed_557_{}/model.pt'.format(l)))

        dataset = prepare_data(args, 'data/IBMDebate/filtered/ibm_test_prompts.json', tokenizer, l, source='IBM')
        save_path = 'flan-t5-prompted/ibm-results/'
        print("test start...")
        #evaluate model
        generate_output(args, tokenizer, model, dataset, save_path, l)