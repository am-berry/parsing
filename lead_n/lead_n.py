from rouge import Rouge
import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
import json

# Load test set into memory 
test = pd.read_csv('../data/test.csv')

# Split full text into sentences 
test['tokenized_text'] = test['Text'].apply(nltk.sent_tokenize)
test['tokenized_sum'] = test['Sum'].astype(str).apply(nltk.sent_tokenize) 

# Calculate length 
test['sum_len'] = test['tokenized_sum'].apply(len)
lead_n = test.apply(lambda x: x.tokenized_text[:x.sum_len], axis= 1)
lead_n = lead_n.apply(lambda x: ''.join([str(a) for a in x]))

# Some text processing
gens = lead_n.tolist()
refs = test['Sum'].tolist()

gen_ref = zip(gens, refs)
gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
gens, refs  = zip(*gen_ref)

gens = [str(g) for g in gens]
refs = [str(r) for r in refs]

# Compute rouge scores 
rouge = Rouge()
scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(scores)

# Write results to csv 
results_df = pd.DataFrame({'references':refs, 'generated':gens})
results_df.to_csv('../results/lead_n_results.csv')

with open('./rouge_results/lead_n.json', 'w') as handler:
    json.dump(scores, handler)

