import rouge 
import nltk
import pandas as pd

df = pd.read_csv('../data/test.csv')

#ignoring the first sentence, then taking n sentences (ONE PLUS LEAD)  

df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)

def lead_p1(x):
    if len(x.tokenized_text) <= 1: 
        return x.tokenized_text
    return x.tokenized_text[1:x.sum_len+1]

df['lead'] = df.apply(lambda x: lead_p1(x), axis=1)

opl_gens = df['lead'].apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()

# tail-n summaries 

df['tail'] = df.apply(lambda x: x.tokenized_text[-x.sum_len:], axis = 1)
tail_gens = df['tail'].apply(lambda x: ''.join([str(a) for a in x])).tolist()

rouge = rouge.Rouge()
opl_scores = rouge.get_scores(opl_gens, refs, avg=True, ignore_empty=True)
tail_scores = rouge.get_scores(tail_gens, refs, avg=True, ignore_empty=True)

print(opl_scores)
print(tail_scores)

import json
with open('./opl_n.json', 'w') as handler:
    json.dump(opl_scores, handler)

with open('./tail_n.json', 'w') as handler:
    json.dump(tail_scores, handler)
