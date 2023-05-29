import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from copy import deepcopy
import json
from tqdm import tqdm
from chaosnli_config import *
from chaosnli_dataloader import *
import random, os

class nli_task(pl.LightningModule):
    
    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]
        self.args = args


    def training_step(self, batch, batch_idx):
        self.model.train()
        if 'bert' in self.args['model_name'] and not 'roberta' in self.args['model_name']:
            #print(batch)
            # result = pl.TrainResult(loss)
            # result.log('train_loss', loss, on_epoch=True)
            loss = self.model(batch["input_ids"],attention_mask = batch["attention_mask"],token_type_ids = batch["token_type_ids"], labels=batch['label']).loss
        if 'roberta' in self.args['model_name']:
            loss = self.model(batch['input_ids'], attention_mask = batch['attention_mask'], labels = batch['label']).loss

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
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

def train(args, *more):
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"] + str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    seed_everything(args["seed"])


    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bert" in args["model_name"] and not "roberta" in args["model_name"]:
        model = BertForSequenceClassification.from_pretrained(args["model_checkpoint"], num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(args["model_checkpoint"])
        model.config.id2label = {'0':'entailment', '1':'neutral', '2':'contradiction'}
        model.config.label2id = {'entailment':0, 'neutral':1, 'contradiction':2}
    elif 'roberta' in args["model_name"]:
        model = RobertaForSequenceClassification.from_pretrained(args['model_checkpoint'], num_labels=3)
        tokenizer = RobertaTokenizer.from_pretrained(args['model_checkpoint'])
        model.config.id2label = {'0':'entailment', '1':'neutral', '2':'contradiction'}
        model.config.label2id = {'entailment':0, 'neutral':1, 'contradiction':2}
        

    task = nli_task(args, tokenizer, model)

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

    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    evaluate_model(args, task.tokenizer, task.model, test_loader, save_path)

def evaluate_model(args, tokenizer, model, test_loader, save_path):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    
    loss = 0
    counter = 0
    model.to('cuda')
    for batch in tqdm(test_loader):
        with torch.no_grad():
            #print(batch)
            if 'bert' in args['model_name']:
                if 'roberta' in args['model_name']:
                    print(batch['input_ids'].shape, batch['attention_mask'].shape)
                    logits = model(batch["input_ids"].to(device='cuda'),attention_mask = batch["attention_mask"].to(device='cuda')).logits
                    loss += kl_loss(torch.nn.functional.log_softmax(logits), batch['label'].to(device='cuda'))
                    counter += 1
                    print(counter)

    
    print('KL Divergence: {}'.format(str(loss/counter)))
    with open(save_path + 'results.json', 'w') as f:
        f.write('KL Divergence: {}'.format(str(loss/counter)))
        f.close()

if __name__ == '__main__':
    args = get_args()
    if args.mode=="train":
        train(args)
    if args.mode=='test':
        args = vars(args)
        # hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
        # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
        hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

        tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
        model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
        model.cuda()
        task = nli_task(args, tokenizer, model)

        train_loader, val_loader, test_loader = prepare_data(args, task.tokenizer)
        save_path = 'roberta-raw'
        evaluate_model(args, task.tokenizer, task.model, val_loader, save_path)