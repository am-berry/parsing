# Script for randomly sampling examples for appendices 

import pandas as pd

df = pd.read_csv('finetuned_beam4.csv')

print(df.sample(1).values.tolist())
