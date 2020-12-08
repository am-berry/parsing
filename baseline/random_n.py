import random
import numpy as np
import pandas as pd
import nltk
import rouge

nltk.download('punkt')

df = pd.read_csv('../data/test.csv')

df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)

def get_n_random(x):
    return random.sample(x.tokenized_text, k=x.sum_len) 

rands = df.apply(lambda x: get_n_random(x), axis=1)
gens = rands.apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()

rouge = rouge.Rouge()
r_scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(r_scores)

import json
with open('./random.json', 'w') as handler:
    json.dump(r_scores, handler)
