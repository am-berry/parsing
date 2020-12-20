import rouge 
import nltk
import pandas as pd
import string
import json 

# Loading test set into memory 
df = pd.read_csv('../data/test.csv')

# Tokenizing full text, calculating summary length 
df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)
df['tokenized_summ'] = df['Sum'].astype(str).apply(nltk.sent_tokenize)
df['sum_len'] = df['tokenized_summ'].apply(len)

####### 1+lead-n #######

tok = df['tokenized_text'].tolist()
t = [x[1:] for x in tok]
df['moved_tokenized_text'] = t

df['lead'] = df.apply(lambda x: x.moved_tokenized_text[:x.sum_len], axis=1)

opl_gens = df['lead'].apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()

gen_ref = zip(opl_gens, refs)
gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
opl_gens, opl_refs  = zip(*gen_ref)

# tail-n summaries 

df['tail'] = df.apply(lambda x: x.tokenized_text[-x.sum_len:], axis = 1)
tail_gens = df['tail'].apply(lambda x: ''.join([str(a) for a in x])).tolist()

gen_ref = zip(tail_gens, refs)
gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
tail_gens, tail_refs = zip(*gen_ref)

# String operations 
opl_refs = [str(r) for r in opl_refs]
opl_gens = [str(o) for o in opl_gens]
tail_gens = [str(g) for g in tail_gens]
tail_refs = [str(a) for a in tail_refs]

rouge = rouge.Rouge()
opl_scores = rouge.get_scores(opl_gens, opl_refs, avg=True, ignore_empty=True)
tail_scores = rouge.get_scores(tail_gens, tail_refs, avg=True, ignore_empty=True)

print(opl_scores)
print(tail_scores)

# Write to csv
results_df = pd.DataFrame({'references':tail_refs, 'generated':tail_gens})
results_df.to_csv('../results/tail_results.csv')
results_df = pd.DataFrame({'references':opl_refs, 'generated':opl_gens})
results_df.to_csv('../results/tail_results.csv')

# Write scores to json
with open('./rouge_results/opl_n.json', 'w') as handler:
    json.dump(opl_scores, handler)

with open('./rouge_results/tail_n.json', 'w') as handler:
    json.dump(tail_scores, handler)
