import json, os, re

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
    labels = ['entailment', 'contradiction', 'neutral']
    total = 0
    total_matched = 0
    total_token = 0
    micro_recall = 0
    for l in labels:
        gold = 'contrastive_augmented_snli_{}.json'.format(l)
        test_folder = 'save/t5-smallt5-small0.0001_epoch_5_seed_557_{}/results/'.format(l)
        test_file = 'results_{}.json'.format(l)

        with open(gold, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        
        print(len(gold_data))

        with open(test_folder+test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(len(test_data))

        for i in range(len(gold_data)):
            tokens = find_highlighted(gold_data[i]['contrastive_highlight'])#(gold_data[i]['highlighted_premise'] + gold_data[i]['highlighted_hypothesis'])
            a, b = get_recall(tokens, test_data[i]['explanation'].lower())
            total +=1
            total_matched += a
            total_token += b
            micro_recall += a/b

        with open(os.path.join(test_folder,'auto_eval_results__augmented_{}.txt'.format(l)), 'w') as f:
            f.write('Total matched:{}, total token:{} in {} instances. \n'.format(str(total_matched), str(total_token), str(total)))
            f.write('Macro Recall:{}, Micro Recall:{}. \n'.format(str(total_matched/total_token), str(micro_recall/total)))
            f.close()