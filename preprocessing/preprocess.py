#!/usr/bin/env python3

import pandas as pd
import nltk
import rouge

'''
d = pd.read_csv('res.csv') 

def f(x):
    return x[6:]

def g(x):
    step = 0
    try:
        while not x[step].isalpha():
            step += 1
    except:
        return x 
    return x[step:]

def h(x):
    return x.lstrip()

d['Sum'] = d['Summary'].apply(f)
d['Sum'] = d['Sum'].apply(g)
d['Sum'] = d['Sum'].apply(h)

print(d.shape)

d.drop_duplicates(inplace=True)
d.drop('Summary', axis= 1,inplace=True)

d.to_csv('processed.csv', sep = ',', index=False)
'''
nltk.download('punkt')

d = pd.read_csv('processed.csv', sep=',', error_bad_lines=False)
print(d['Sum'].isna().sum())
d.dropna(inplace=True)
print(d['Sum'].isna().sum())
d['sum_sents'] = d['Sum'].apply(nltk.sent_tokenize)
d['sum_len'] = d['sum_sents'].apply(len)

print(d.head(10))
d.to_csv('prp.csv', sep=',', index=False)



