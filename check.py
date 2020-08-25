#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('res.csv') 
print(d.head(10))

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

print(d.head())
print(d.describe())
