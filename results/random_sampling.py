# Script for randomly sampling examples for appendices 

import pandas as pd

df = pd.read_csv('full_lexrank.csv')

print(df.sample(3).values.tolist())
