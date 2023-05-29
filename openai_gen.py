import openai
import os, json
from tqdm import tqdm
import backoff  # for exponential backoff

openai.api_key = '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'

def get_item(data, label):
        if label == 'entailment':
            prompt = 'Explain why the premise entails the hypothesis. '
        elif label == 'contradiction':
            prompt = 'Explain why the premise contradicts the hypothesis. '
        elif label == 'neutral':
            prompt = 'Explain why the premise is neutral with respect to the hypothesis. '

        x = {}

        x['input_text'] = prompt + 'Premise: ' + data['premise'] + ' ' + 'Hypothesis: ' +  data['hypothesis']
        x['premise'] = data['premise']
        x['hypothesis'] = data['hypothesis']
        if 'explanation' in data.keys():
            x['output_text'] = 'Explanation: ' + data['explanation']
            x['whole_sequence'] = x['input_text'] + ' ' + x['output_text']

        return x

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


if __name__ == '__main__':
    in_context_tr = {}
    new_d = []
    labels = ['contradiction', 'entailment', 'neutral']
    for l in labels:
        path_train = 'data/eSNLI/train_e-snli-{}.json'.format(l)
        with open(path_train, 'r', encoding='utf-8') as f:
            train = json.load(f)
        tr = get_item(train[0], l) #in-context
        in_context_tr[l] = tr

    with open('data/chaosNLI/chaosNLI_snli.json', 'r') as f:
        chaosNLI = json.load(f)

    for d in tqdm(chaosNLI[:100], desc='chaosNLI'):
        for l in labels:
            t = get_item(d['example'], l)
            response = completions_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "user", "content": in_context_tr[l]['input_text']},
                        {"role": "assistant", "content": in_context_tr[l]['output_text']},
                        {"role": "user", "content": t['input_text']}
                    ]
                )
            exp = response['choices'][0]['message']['content']
            d['example']['{}_explanation'.format(l)] = exp
            new_d.append(d)

    with open('gpt3.5_results/chaosNLI_snli_augmented.json','w') as f:
        f.write(json.dumps(new_d, indent=2))
        f.close()