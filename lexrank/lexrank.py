#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
import numpy as np
from lexrank_utils import *
import time
import pickle

t = time.time()

test = pd.read_csv('../data/test.csv', sep=',')[10:]
print(test[test['sum_len'] == 0].sum())
model = SentenceTransformer('../models/', device='cuda')

test['split_text'] = test['Text'].apply(nltk.sent_tokenize)
split_sents = test['split_text'].tolist()

embeddings = []
for i, doc in enumerate(split_sents):
    if i % 100 == 0:
        print(f'{i} sentences encoded')
    embeddings.append(model.encode(doc))

print(f'embeddings done')
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

print('centrality started')
centrality = []
for i, cos in enumerate(pairwise_cos):
    if i % 1 == 0:
        print(f'{i} centralities computed')
    scores = degree_centrality_scores(cos)
    centrality.append(np.argsort(-scores))

with open('./centrality.pkl', 'wb') as handler:
    pickle.dump(centrality, handler)

print('centrality done')
del pairwise_cos

sum_len = test['sum_len'].tolist()
zipped = zip(sum_len, centrality)
c = [pair[1][:pair[0]] for pair in zipped]

def sample(lst, indices):
    return [lst[i] for i in indices]

test['ix'] = c
test['generated'] = test.apply(lambda row: sample(row['split_text'], row['ix']), axis =1)
print(f'{time.time()-t}')

test['joined'] = test['generated'].apply(lambda x: ' '.join([str(a) for a in x]))
test[['Text', 'Sum', 'joined']].to_csv('lexrank_results.csv', index=False, sep=',')

from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(test['joined'].tolist(), test['Sum'].tolist(), avg=True, ignore_empty=True)
import json
with open('../results/lexrank.json', 'w') as handler:
        json.dump(scores, handler)

print(scores)
