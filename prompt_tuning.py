import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from copy import deepcopy
import json
from tqdm import tqdm
from config import *
from data_loader import *
import random, os, json, re

class Soft_Prompt_T5(nn.Module):
    def __init__(self, model, tokenizer):
        super(Soft_Prompt_T5, self).__init__()  
        self.t5 = model
        self.t5.eval()
        #self.config = config #model config
        self.tokenizer= tokenizer
        self.soft_embedding_layer=None   
        self.normal_embedding_layer=self.t5.get_input_embeddings()
        self.n_tokens = 10 #length of soft-prompt
            
        
        #[3,27569,10],[11167,10],[31484,17,10,1]
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(self.normal_embedding_layer, n_tokens=self.n_tokens))    

    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5, initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.normal_embedding_layer.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
        

    def prepare_input(self, input_ids,attention_mask):
        batch_size=input_ids.shape[0]
        decoder_input_ids=torch.zeros(batch_size,1,dtype=int).to(input_ids.device)
        
        input_embeddings=self.normal_embedding_layer(input_ids)
        prefix_soft_embeddings = self.learned_embedding.repeat(input_embeddings.size(0), 1, 1)
        
        input_embeddings=torch.cat(
            [prefix_soft_embeddings, input_embeddings],
            dim=1
            )
        
        prefix_soft_attention_mask = torch.ones(batch_size,self.n_tokens).to(input_ids.device)
        attention_mask=torch.cat(
            [prefix_soft_attention_mask, attention_mask],
            dim=1
            )
        
        return input_embeddings, attention_mask
    def forward(self,input_ids,attention_mask,labels):

        input_embeddings, attention_mask = self.prepare_input(input_ids, attention_mask)

        return self.t5(
                    inputs_embeds=input_embeddings,
                    labels=labels,
                    attention_mask=attention_mask,
                    return_dict=True
                )
    
    def generate(self,input_ids, attention_mask, eos, length):
        with torch.no_grad():
            input_embeddings, attention_mask= self.prepare_input(input_ids, attention_mask)

            outputs = self.t5.generate(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=length
                    )
        return outputs

class prompt_exp_task(pl.LightningModule):
    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = Soft_Prompt_T5(model, tokenizer)
        self.lr = args["lr"]
        self.args = args

    def training_step(self, batch, batch_idx):
        model = self.model
        model.learned_embedding.requires_grad_(True)

        if 'bert' in self.args['model_name'] and not 'roberta' in self.args['model_name']:
            #print(batch)
            # result = pl.TrainResult(loss)
            # result.log('train_loss', loss, on_epoch=True)
            loss = self.model(batch["input_ids"],attention_mask = batch["attention_mask"],token_type_ids = batch["token_type_ids"], labels=batch['label']).loss
        if 'roberta' in self.args['model_name']:
            loss = self.model(batch['input_ids'], attention_mask = batch['attention_mask'], labels = batch['label']).loss
        elif 't5' in self.args['model_name']:
            loss = self.model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["label_ids"]).loss
        elif 'bloom' in self.args['model_name']:
            loss = self.model(input_ids=batch["sequence_ids"],
                            labels=batch["sequence_ids"]).loss
        self.log('train_loss', loss)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()        
        if 'bert' in self.args['model_name'] and not 'roberta' in self.args['model_name']:
            #print(batch)
            # result = pl.TrainResult(loss)
            # result.log('train_loss', loss, on_epoch=True)
            loss = self.model(batch["input_ids"],attention_mask = batch["attention_mask"],token_type_ids = batch["token_type_ids"], labels=batch['label']).loss
        if 'roberta' in self.args['model_name']:
            # print(batch['input_ids'].shape)
            # print(batch['attention_mask'].shape)
            # print(batch['label'].shape)
            loss = self.model(batch['input_ids'], attention_mask = batch['attention_mask'], labels = batch['label']).loss
        elif 't5' in self.args['model_name']:
            loss = self.model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["label_ids"]).loss
        elif 'bloom' in self.args['model_name']:
            loss = self.model(input_ids=batch["input_ids"],
                            labels=batch["sequence_ids"]).loss
        #print(loss)
        self.log('val_loss', loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        #print(outputs[0]['val_loss'])
        #print(outputs)
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()}, 'val_loss': val_loss_mean.item()}
        self.log("val_loss", results['val_loss'])
        return results

    def configure_optimizers(self):
        return AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, correct_bias=True)

def train(args, *more):
    args = vars(args)
    args["model_name"] = args['data'] + args["model_name"] + str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + '_' + args['label']
    # train!
    seed_everything(args["seed"])


    if "t5" in args["model_name"]:
        language_model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        language_model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif 'bloom' in args["model_name"]:
        model = BloomForCausalLM.from_pretrained(args["model_checkpoint"])
        tokenizer = AutoTokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    # elif "bert" in args["model_name"] and not "roberta" in args["model_name"]:
    #     model = BertForSequenceClassification.from_pretrained(args["model_checkpoint"], num_labels=3)
    #     tokenizer = BertTokenizer.from_pretrained(args["model_checkpoint"])
    #     model.config.id2label = {'0':'entailment', '1':'neutral', '2':'contradiction'}
    #     model.config.label2id = {'entailment':0, 'neutral':1, 'contradiction':2}
    # elif 'roberta' in args["model_name"]:
    #     model = RobertaForSequenceClassification.from_pretrained(args['model_checkpoint'], num_labels=3)
    #     tokenizer = RobertaTokenizer.from_pretrained(args['model_checkpoint'])
    #     model.config.id2label = {'0':'entailment', '1':'neutral', '2':'contradiction'}
    #     model.config.label2id = {'entailment':0, 'neutral':1, 'contradiction':2}
        
    for param in language_model.parameters():
            param.requires_grad = False
    task = prompt_exp_task(args, tokenizer, language_model)
    

    train_loader, val_loader, test_loader = prepare_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=2,verbose=False, mode='min')],
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    #precision=16,
                    accelerator="cuda"
                    )

    trainer.fit(task, train_loader, val_loader)


    torch.save(task.model.state_dict(), os.path.join(save_path, 'model.pt'))
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    evaluate_model(args, task.tokenizer, task.model, test_loader, save_path)

def evaluate_model(args, tokenizer, model, test_loader, save_path):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save = []
    auto_eval = args['auto_eval']
    total = 0
    total_matched = 0
    total_token = 0
    micro_recall = 0

    model.to('cuda')
    for batch in tqdm(test_loader):
        with torch.no_grad():
            model.cuda()
            if 't5' in args['model_name']:
                #print(batch)
                outputs = model.generate(batch["input_ids"].to(device='cuda'),batch["attention_mask"].to(device='cuda'), tokenizer.eos_token_id, 128)
            elif 'bloom' in args['model_name']:
                outputs = model.generate(input_ids=batch["input_ids"].to(device='cuda'),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=128
                                )
            
            outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(outputs_text)
            # print(batch)

            for idx in range(len(outputs_text)):
                tmp_save = {'premise': batch['premise'][idx], 'hypothesis': batch['hypothesis'][idx], 'gold_explanation':batch['output_text'][idx]}
                tmp_save['explanation'] = outputs_text[idx]
                tmp_save['input_text'] = batch['input_text'][idx]
                save.append(tmp_save)

            if auto_eval == True:
                highlighted_tokens = find_highlighted(batch['highlighted_premise'][idx] + batch['highlighted_hypothesis'][idx])
                matched_token, token_num = get_recall(highlighted_tokens, tmp_save['explanation'].lower())
                total +=1
                total_matched += matched_token
                total_token += token_num
                micro_recall += matched_token/token_num
    
    with open(os.path.join(save_path,'results_{}.json'.format(args['label'])), 'w') as f:
        f.write(json.dumps(save, indent=2))
        f.close()

    if auto_eval == True:
        with open(os.path.join(save_path,'auto_eval_results_{}.txt'.format(args['label'])), 'w') as f:
            f.write('Total matched:{}, total token:{} in {} instances. \n'.format(str(total_matched), str(total_token), str(total)))
            f.write('Macro Recall:{}, Micro Recall:{}. \n'.format(str(total_matched/total_token), str(micro_recall/total)))
            f.close()

def find_highlighted(t):
    puncts = '!@#$%^&*()_+-=`~[]\{\};:",./<>?\''
    highlighted = re.findall(r'\*(.*?)\*',t)
    lowered = set()
    for token in highlighted:
        for i in range(len(puncts)):
            token = token.replace(puncts[i], "")
        lowered.add(token.lower())
    return lowered

def get_recall(tokens, exp):
    matched = 0
    for t in tokens:
        if t in exp:
            matched += 1
    return matched, len(tokens)

if __name__ == '__main__':
    args = get_args()
    if args.mode=="train":
        train(args)
