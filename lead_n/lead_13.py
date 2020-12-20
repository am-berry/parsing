import rouge 
import nltk
import pandas as pd
import json

df = pd.read_csv('../data/test.csv')

# Split text into sentences
df['tokenized_text'] = df['Text'].apply(nltk.sent_tokenize)

# Return first sentence (lead-1)
lead_1 = df['tokenized_text'].apply(lambda x: x[:1])

# Return first three sentences (lead-3)
lead_3 = df['tokenized_text'].apply(lambda x: x[:min(len(x), 3)])

# Fixing the data for input into rouge library
df['lead_1'] = lead_1.apply(lambda x: ''.join([str(a) for a in x]))
df['lead_3'] = lead_3.apply(lambda x: ''.join([str(a) for a in x]))

refs = df['Sum'].tolist()
l1_gens = df['lead_1'].tolist()
l3_gens = df['lead_3'].tolist()

refs = [str(x) for x in refs]
l1_gens = [str(a) for a in l1_gens]
l3_gens = [str(b) for b in l3_gens]

# Calculating ROUGE scores for both 
rouge = rouge.Rouge()
l1_scores = rouge.get_scores(l1_gens, refs, avg=True, ignore_empty=True)
l3_scores = rouge.get_scores(l3_gens, refs, avg=True, ignore_empty=True)

# Writing the generated summaries to csv file 
df['lead1'] = l1_gens
df['lead3'] = l3_gens 

df = df[['Text', 'Sum', 'lead1', 'lead3']]
df.to_csv('../results/lead1and3.csv')

# Sending the resulting rouge scores to json files 
with open('./rouge_results/lead_1.json', 'w') as handler:
    json.dump(l1_scores, handler)

with open('./rouge_results/lead_3.json', 'w') as handler:
    json.dump(l3_scores, handler)
