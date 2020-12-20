import pandas as pd 
import nltk
from sklearn.model_selection import train_test_split

r = 17 

# Load preprocessed csv into memory 
df = pd.read_csv('../data/prp.csv', sep = ',')

# Split into train, test slightly larger than wanted. Fix random seed for reproducibility 
train, test = train_test_split(df, train_size=120000, test_size=60000, shuffle=True, random_state=r)

# Clear out broken values 
train.dropna(inplace=True)
test.dropna(inplace=True)

# Only keep examples where length of the summary is between 25 and 500 characters
train_df = train[train['Sum'].apply(lambda x: 500 > len(x) > 25)]
test_df = test[test['Sum'].apply(lambda x: 500 > len(x) > 25)]

# Write to file 
train_df.to_csv('../data/train.csv', sep=',', index=False)
test_df.to_csv('../data/test.csv', sep=',', index=False)
