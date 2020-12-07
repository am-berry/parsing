import pandas as pd

# looking at some random examples 

pd.options.display.width = 0
df = pd.read_csv('./data/prp.csv')
print(df.sample(n=3, random_state=7).to_string())
