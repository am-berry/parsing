import random
import string

import numpy as np
import pandas as pd
import nltk
import rouge

df = pd.read_csv('../data/test.csv')

df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)

rands = df['tokenized_text'].apply(lambda x: random.sample(x, min(3, len(x))))
gens = rands.apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()
print(len(gens), len(refs))

gen_ref = zip(gens, refs)
gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
gens, refs  = zip(*gen_ref)
print(len(gens), len(refs))
 
def get_n_random(x):
    return random.sample(x.tokenized_text, k=x.sum_len) 

rands = df.apply(lambda x: get_n_random(x), axis=1)
gens = rands.apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()

rouge = rouge.Rouge()
r_scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(r_scores)

import json
with open('./random3.json', 'w') as handler:
    json.dump(r_scores, handler)
