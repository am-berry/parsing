#!/usr/bin/env python3

# This is to get reddit data in the format of the CNN/DM for easy use later.

# Step 1 - making .story files 

import pandas as pd

d = pd.read_csv('processed.csv', error_bad_lines=False)

print(d.columns)

for i, row in d.iterrows():
    print(i)
    txt = row['Text']
    summ = row['Sum']
    try:
        p1 = ('.\n'.join(txt.split('.')))
        summ_sents = "@highlight\n"
        for sent in summ.split('.'):
            summ_sents += sent 
            summ_sents += "\n@highlight\n"
        fin = p1+'\n'+summ_sents
        fin =\
        fin.strip().rstrip('\n@highlight\n').rstrip('**').strip().rstrip('@highlight').strip()
        with open(f'./raw/{i}.story', "w") as f:
            f.write(fin)
    except:
        continue
