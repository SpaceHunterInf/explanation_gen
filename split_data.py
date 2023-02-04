import os, json

for filename in os.listdir('data/eSNLI/'):
    with open(os.path.join('data/eSNLI', filename), 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(os.path.join('data/eSNLI', 'train_' + filename), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data[:int(0.8*len(data))], indent=2))
        f.close()
    
    with open(os.path.join('data/eSNLI', 'dev_' + filename), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data[int(0.8*len(data)):int(0.9*len(data))], indent=2))
        f.close()
    
    with open(os.path.join('data/eSNLI', 'test_' + filename), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data[int(0.9*len(data)):], indent=2))
        f.close()
    