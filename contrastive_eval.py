from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np
import os, json
from exp_test import get_recall
from copy import deepcopy
from tqdm import tqdm

def all_masks(tokenized_text):
    # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    # WITHOUT empty and full sets!
    s = list(range(len(tokenized_text)))
    x = len(s)
    masks = [1 << i for i in range(x)]
    #     for i in range(1 << x):  # empty and full sets included here
    for i in range(1, 1 << x - 1):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
        
def all_consecutive_masks(tokenized_text, max_length = -1):
    # WITHOUT empty and full sets!
    s = list(range(len(tokenized_text)))
    x = len(s)
    for i in range(x):
        for j in range(i+1, x):
            mask = s[:i] + s[j:]
            if max_length > 0:
                if j - i >= max_length:
                    yield mask
            else:
                yield mask
                
def all_consecutive_masks2(tokenized_text, max_length = -1):
    # WITHOUT empty and full sets!
    s = list(range(len(tokenized_text)))
    x = len(s)
    for i in range(x+1):
        for j in range(i+1, x+1):
            mask = s[i:j]
            if max_length > 0:
                if j - i <= max_length:
                    yield mask
            else:
                yield mask

def predict_json(ex, model, tokenizer):
    premise = ex['premise']
    hypothesis = ex['hypothesis']
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).cuda()
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).cuda()
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).cuda()

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    
    label = model.config.id2label[torch.argmax(outputs[0]).item()]
    
    return {'encoded_representations': encoding.cpu(), 'label':label}


def get_masked_encodings(predicted, model, tokenizer, fact, foil):

    ex = deepcopy(predicted)
    out = predict_json(ex, model, tokenizer)
    encoded_orig = out['encoded_representations']

    # assert fact != foil, "Fact should be different from the foil (if not, pick a different foil)"
    #print(ex)
    ex['premise'] = ex['premise'].split()
    ex['hypothesis'] = ex['hypothesis'].split()

    #tokenizer.convert_tokens_to_string(out['tokens'])

    masks1 = [[]]  # change this if you also want to mask out parts of the premise.
    masks2 = list(all_consecutive_masks2(ex['hypothesis'], max_length=1))
    encoded = []
    mask_mapping = []
    preds = np.zeros(shape=(len(masks1), len(masks2)))

    for m1_i, m1 in enumerate(masks1):
        masked1 = list(ex['premise'])
        for i in m1:
            masked1[i] = tokenizer.mask_token
        masked1 = ' '.join(masked1)
            
        for m2_i, m2 in enumerate(masks2):
            masked2 = list(ex['hypothesis'])
            for i in m2:
                masked2[i] = tokenizer.mask_token
            masked2 = ' '.join(masked2)
                
            masked_ex = {
                "premise": masked1,
                "hypothesis": masked2
            }
            
            masked_out = predict_json(masked_ex, model, tokenizer)
    #         if masked_out['label'] != foil:
    #             continue
            
            #print(m1_i, m2_i)
            #print(f"{masked1}\n{masked2}")
            #print(masked_ex)
            #print(masked_out['label'])
            encoded.append(masked_out['encoded_representations'].detach().numpy())
            mask_mapping.append((m1_i, m2_i))
            
            #print("====")
    encoded = np.array(encoded)
    return encoded_orig, encoded, mask_mapping, masks1, masks2

def get_highlighted_ranking(encoded_orig, encoded, contrastive_projection):
    z_all = encoded_orig 
    z_h = encoded 
    z_all_row = encoded_orig @ contrastive_projection
    z_h_row = encoded @ contrastive_projection

    prediction_probabilities = softmax(z_all_row @ classifier_w.T + classifier_b)
    prediction_probabilities = np.tile(prediction_probabilities, (z_h_row.shape[0], 1))

    prediction_probabilities_del = softmax(z_h_row @ classifier_w.T + classifier_b, axis=1).squeeze(1)
    #print(prediction_probabilities.shape)
    #print(prediction_probabilities_del.shape)
    p = prediction_probabilities[:, [fact_idx, foil_idx]]
    q = prediction_probabilities_del[:, [fact_idx, foil_idx]]

    p = p / p.sum(axis=1).reshape(-1, 1)
    q = q / q.sum(axis=1).reshape(-1, 1)
    distances = (p[:, 0] - q[:, 0])

    #print("=========\n=======Farthest masks:=======")
        
    highlight_rankings = np.argsort(-distances)
    return highlight_rankings

if __name__ == '__main__':
    max_length = 256

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = RobertaTokenizer.from_pretrained(hg_model_hub_name)
    model = RobertaForSequenceClassification.from_pretrained(hg_model_hub_name)
    model.cuda()
    
    matrix = None #linear classifier out_proj
    bias = None #linear classifier bias
    dense_m = None #dense aka pooling
    dense_b = None #dense bias

    for name, param in model.named_parameters():
        if name == 'classifier.dense.weight':
            dense_m = param.cpu()
        if name == 'classifier.dense.bias':
            dense_b = param.cpu()
        if name == 'classifier.out_proj.weight':
            matrix = param.cpu()
        if name == 'classifier.out_proj.bias':
            bias = param.cpu()


    labels = ['entailment', 'contradiction', 'neutral']
    label2index = {'entailment':0, 'neutral':1, 'contradiction':2}
    fact = 'neutral'
    contrastive_highlighted = []
    with open('save/t5-smallt5-small0.0001_epoch_5_seed_557_{}/results/results_{}.json'.format(fact, fact), 'r', encoding='utf-8') as f:
        predicted_data = json.load(f)

    for predicted in tqdm(predicted_data, desc='data processing'):
        premise = predicted['premise']
        hypothesis = predicted['hypothesis']

        tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                        max_length=max_length,
                                                        return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).cuda()
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).cuda()
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).cuda()

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True,
                        labels=None)
        
        # Note:
        # "id2label": {
        #     "0": "entailment",
        #     "1": "neutral",
        #     "2": "contradiction"
        # },
        #print(outputs[0].shape)
        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

        last_hidden = outputs[1][-1][:,0,:].cpu() # taking only the [CLS] token
        dense_d = torch.mm(last_hidden, dense_m.T) + dense_b
        dense_d = torch.tanh(dense_d)
        encoding = dense_d #encoding after pooling? dense? 1*1024

        highlighted_tokens = set()
        for l in labels:
            if l != fact:
                foil  = l
                encoded_orig, encoded, mask_mapping, masks1, masks2 = get_masked_encodings(predicted, model, tokenizer, fact, foil)

                fact_idx = label2index[fact]
                foil_idx = label2index[foil]
                index2label = model.config.id2label
                #print('fact:', index2label[fact_idx])
                #print('foil:', index2label[foil_idx])
                num_classifiers = 100

                classifier_w = matrix.clone().detach().numpy()
                classifier_b = bias.clone().detach().numpy()

                u = classifier_w[fact_idx] - classifier_w[foil_idx]
                contrastive_projection = np.outer(u, u) / np.dot(u, u)

                #print(contrastive_projection.shape)
                highlighted_rankings = get_highlighted_ranking(encoded_orig.clone().detach(), encoded, contrastive_projection)

                top_k = 4
                if len(highlighted_rankings) < top_k:
                    top_k = len(highlighted_rankings)
                for i in range(top_k):
                    rank = highlighted_rankings[i]
                    m1_i, m2_i = mask_mapping[rank]
                    
                    masked1 = list(predicted['premise'].split())
                    for k in masks1[m1_i]:
                        highlighted_tokens.add(masked1[k].lower())
                    
                    
                    masked2 = list(predicted['hypothesis'].split())
                    for k in masks2[m2_i]:
                        highlighted_tokens.add(masked2[k].lower())
        
        #print(highlighted_tokens)
        contrastive = {}
        contrastive['premise'] = predicted['premise']
        contrastive['hypothesis'] = predicted['hypothesis']
        contrastive['contrastive_highlight'] = ' '.join(['*'+x+'*' for x in highlighted_tokens])
        contrastive_highlighted.append(contrastive)
    
    with open('contrastive_augmented_{}_{}.json'.format('snli', fact), 'w', encoding='utf-8') as f:
        f.write(json.dumps(contrastive_highlighted, indent=2))
        f.close()


                



            

