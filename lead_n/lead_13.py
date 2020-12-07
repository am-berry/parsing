import rouge 
import nltk
import pandas as pd

df = pd.read_csv('../data/test.csv')

# lead 1 

df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)
lead_1 = df['tokenized_text'].apply(lambda x: x[:1])

# lead 3

def up_to_3(x):
    if len(x) < 3:
        return x[:len(x)]
    return x[:3]

lead_3 = df['tokenized_text'].apply(lambda x: up_to_3(x))

df['lead_1'] = lead_1.apply(lambda x: ''.join([str(a) for a in x]))
df['lead_3'] = lead_3.apply(lambda x: ''.join([str(a) for a in x]))

refs = df['Sum'].tolist()
l1_gens = df['lead_1'].tolist()
l3_gens = df['lead_3'].tolist()

rouge = rouge.Rouge()
l1_scores = rouge.get_scores(l1_gens, refs, avg=True, ignore_empty=True)
l3_scores = rouge.get_scores(l3_gens, refs, avg=True, ignore_empty=True)

print(l1_scores)
print(l3_scores)

import json
with open('./lead_1.json', 'w') as handler:
    json.dump(l1_scores, handler)

with open('./lead_3.json', 'w') as handler:
    json.dump(l3_scores, handler)
