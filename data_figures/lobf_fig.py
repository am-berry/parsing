import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/test.csv')

df['tokenized_words'] = df['Text'].astype(str).apply(nltk.word_tokenize)
df['len_text_words'] = df['tokenized_words'].apply(len)
df['tokenized_summ'] = df['Sum'].astype(str).apply(nltk.word_tokenize)
df['len_sum_words'] = df['tokenized_summ'].apply(len)
df.dropna(inplace=True)

ax = sns.regplot(x='len_text_words', y='len_sum_words', data=df)
plt.savefig('lobf.png')
