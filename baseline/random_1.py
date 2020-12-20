import random
import string

import numpy as np
import pandas as pd
import nltk
import rouge
import json 

# Loading test csv in 
df = pd.read_csv('../data/test.csv')

# Tokenizing original text into sentences
df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)

# Sampling 1 sentence randomly from tokenized text. 
rands = df['tokenized_text'].apply(lambda x: random.sample(x, 1))

# Necessary fixing of text for next input into ROUGE 
gens = rands.apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()

gen_ref = zip(gens, refs)
gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
gens, refs  = zip(*gen_ref)

gens = [str(g) for g in gens]
refs = [str(r) for r in refs]

# ROUGE score calculation
rouge = rouge.Rouge()
r_scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)

# Write output to csv file 
results_df = pd.DataFrame({'reference': refs, 'generated': gens})
results_df.to_csv('../results/random_1.csv')

# Write scores to json 
with open('./random.json', 'w') as handler:
    json.dump(r_scores, handler)
