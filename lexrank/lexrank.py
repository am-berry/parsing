#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
import numpy as np
from lexrank_utils import *
import time
import pickle
import gc
import string 

t = time.time()
oa_scores = []

# Load models into memory 
model = SentenceTransformer('../models/roberta-large-ft/')
# Encode a random sentence as the model implementation requires extra time for first encoding for some reason
q = model.encode(["asdf"])

# Load dataset into memory, cleanup 
df = pd.read_csv('../data/test.csv', sep=',')
df.dropna(inplace=True)
df['split_text'] = df['Text'].apply(nltk.sent_tokenize)
df['tokenized_summ'] = df['Sum'].astype(str).apply(nltk.sent_tokenize)
df['sum_len'] = df['tokenized_summ'].apply(len)

# Splitting it into three rounds as mem errors if all done in one go
for a in [(0, 16000), (16000, 32000), (32000, -1)]:
    test = df[a[0]:a[1]] 
    split_sents = test['split_text'].tolist()

    to_embed = []
    shapes = []
    for doc in split_sents:
        to_embed.extend(doc)
        shapes.append(len(doc))

    embedding = model.encode(to_embed)
    print('encoding done')
    embeddings = [] 
    cnt = 0
    for i in shapes:
        embeddings.append(embedding[cnt:cnt+i])
        cnt+=i

    embeddings = np.array(embeddings)
    print(f'embeddings done')
    del embedding
    del split_sents

    # implementation of fast pairwise cosine similarity calculations
    # based on https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    pairwise_cos = []
    for doc in embeddings:
        similarity = np.dot(doc, doc.T)
        square_mag = np.diag(similarity)
        inverse = 1/square_mag
        inverse[np.isinf(inverse)] = 0
        inv_mag = np.sqrt(inverse)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag
        pairwise_cos.append(cosine)

    print('cosine done')

    del embeddings

    # computation of centrality scores imported from lexrank_utils.py 
    print('centrality started')
    centrality = []
    for i, cos in enumerate(pairwise_cos):
        if i % 1 == 0:
            print(f'{i} centralities computed')
        scores = degree_centrality_scores(cos)
        centrality.append(np.argsort(-scores))

    print('centrality done')
    del pairwise_cos

    # Getting the same amount of sentences via lexrank as there is in the reference summary 
    sum_len = test['sum_len'].tolist()
    zipped = zip(sum_len, centrality)
    c = [pair[1][:int(pair[0])] for pair in zipped]

    def sample(lst, indices):
        return [lst[i] for i in indices]

    test['ix'] = c
    test['generated'] = test.apply(lambda row: sample(row['split_text'], row['ix']), axis =1)
    print(f'{time.time()-t}')

    test['joined'] = test['generated'].apply(lambda x: ' '.join([str(a) for a in x]))
    test[['Text', 'Sum', 'joined']].to_csv(f'{a[0]}lexrank_results.csv', index=False, sep=',')

    gens = test['joined'].tolist()
    refs = test['Sum'].tolist()

    gen_ref = zip(gens, refs)
    gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
    gens, refs  = zip(*gen_ref)

    from rouge import Rouge
    rouge = Rouge()
    score = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
    import json
    out_file = f'{a[0]}_lexrank.json'
    with open(out_file, 'w') as handler:
            json.dump(score, handler)

    oa_scores.append(score)
    gc.collect()

print(oa_scores)
