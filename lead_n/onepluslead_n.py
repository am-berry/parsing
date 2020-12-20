import rouge 
import nltk
import pandas as pd

df = pd.read_csv('../data/test.csv')

#ignoring the first sentence, then taking n sentences (ONE PLUS LEAD)  

df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)
df['sum_len'] = df['tokenized_text'].apply(len)
tok = df['tokenized_text'].tolist()

#t = [x[1:] for x in tok]
#df['tokenized_text'] = t

#df['lead'] = df.apply(lambda x: x.tokenized_text[:x.sum_len], axis=1)

#opl_gens = df['lead'].apply(lambda x: ''.join([str(a) for a in x])).tolist()
refs = df['Sum'].tolist()

# tail-n summaries 

df['tail'] = df.apply(lambda x: x.tokenized_text[-x.sum_len:], axis = 1)
tail_gens = df['tail'].apply(lambda x: ''.join([str(a) for a in x])).tolist()

refs = [str(r) for r in refs]
tail_gens = [str(g) for g in tail_gens]

rouge = rouge.Rouge()
#opl_scores = rouge.get_scores(opl_gens, refs, avg=True, ignore_empty=True)
tail_scores = rouge.get_scores(tail_gens, refs, avg=True, ignore_empty=True)

#print(opl_scores)
print(tail_scores)

import json
#with open('./opl_n.json', 'w') as handler:
#    json.dump(opl_scores, handler)

with open('./tail_n.json', 'w') as handler:
    json.dump(tail_scores, handler)
