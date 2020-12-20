from rouge import Rouge
import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
r=17

test = pd.read_csv('../data/test.csv')

test['tokenized_text'] = test['Text'].apply(nltk.sent_tokenize)
test['sum_len'] = test['tokenized_text'].apply(len)
lead_n = test.apply(lambda x: x.tokenized_text[-x.sum_len:], axis = 1)
lead_n = lead_n.apply(lambda x: ''.join([str(a) for a in x]))

gens = lead_n.tolist()
refs = test['Sum'].tolist()
print(len(gens), len(refs))

gen_ref = zip(gens, refs)
gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
gens, refs  = zip(*gen_ref)
print(len(gens), len(refs))

gens = [str(g) for g in gens]
refs = [str(r) for r in refs]

rouge = Rouge()
scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(scores)

import json

with open('./results/tail_n.json', 'w') as handler:
    json.dump(scores, handler)
