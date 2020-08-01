#!/usr/bin/env python

import re
import pandas as pd

d = pd.read_csv('res.csv', dtype=object, index_col = False).transpose()
d = d.reset_index()
d.columns = ['text']

def f(x):
    return re.findall(r'tl;dr(.+ ?)\\n', x)

def g(x):
    return r'{}'.format(x)

d['raw_text'] = d['text'].apply(g)
d['summary'] = d['raw_text'].apply(f)
print(d.columns)
print(d.head())
