#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
import numpy as np
from lexrank_utils import *

test = pd.read_csv('./data/test.csv', sep=',')[20000:20005]
model = SentenceTransformer('./models/')

test['split_text'] = test['Text'].apply(nltk.sent_tokenize)
split_sents = test['split_text'].tolist()

embeddings = []
for doc in split_sents:
    embeddings.append(model.encode(doc))

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

centrality = []
for cos in pairwise_cos:
    scores = degree_centrality_scores(cos)
    centrality.append(scores)

central_indices = []
for score in centrality:
    central_indices.append(np.argsort(-score))

sum_len = test['sum_len'].tolist()
zipped = zip(sum_len, central_indices)
c = [pair[1][:pair[0]] for pair in zipped]

def sample(lst, indices):
    return [lst[i] for i in indices]

test['ix'] = c
test['generated'] = test.apply(lambda row: sample(row['split_text'], row['ix']), axis =1)
