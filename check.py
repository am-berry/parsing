#!/usr/bin/env python3

import re
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('res.csv').transpose() 
d = d.reset_index()
d.columns = ['text']

def f(x):
    return re.findall(r'tl[;:]dr(.+ ?)', x)

d['summary'] = d['text'].apply(f)
d['num_tldr'] = d['summary'].apply(len)

d['summary'] = d['summary'].apply(lambda x: x[0] if len(x)>0 else 0)

d['summary_len'] = d['summary'].apply(lambda x: len(x) if x!=0 else 0)

a = d.hist(column='summary_len', bins = 10)
plt.savefig('i.png')
