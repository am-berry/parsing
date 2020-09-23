#!/usr/bin/env python3

import pandas as pd

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
