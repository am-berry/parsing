#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
import numpy as np
from lexrank_utils import *
import time
import pickle
<<<<<<< HEAD
import gc
import string 

t = time.time()
oa_scores = []

for a in [(30001, -1)]:
=======

t = time.time()
scores = []

for a in [(10,30000), (30001, -1)]:
>>>>>>> 39595f8a87b536d94828777e7220a7fa8dde1ecb
    test = pd.read_csv('../data/test.csv', sep=',')[a[0]:a[1]]
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

<<<<<<< HEAD
    gens = test['joined'].tolist()
    refs = test['Sum'].tolist()

    gen_ref = zip(gens, refs)
    gen_ref = [_ for _ in gen_ref if not all(j in string.punctuation for j in _[0])]
    gens, refs  = zip(*gen_ref)

    from rouge import Rouge
    rouge = Rouge()
    score = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
=======
    from rouge import Rouge
    rouge = Rouge()
    score = rouge.get_scores(test['joined'].tolist(), test['Sum'].tolist(), avg=True, ignore_empty=True)
>>>>>>> 39595f8a87b536d94828777e7220a7fa8dde1ecb
    import json
    out_file = f'{a[0]}_lexrank.json'
    with open(out_file, 'w') as handler:
            json.dump(score, handler)

    print(score)
<<<<<<< HEAD
    oa_scores.append(score)
    gc.collect()

print(oa_scores)
=======
    scores.append(score)

print(scores)
>>>>>>> 39595f8a87b536d94828777e7220a7fa8dde1ecb
