import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../results/full_lexrank.csv')
tokenized_sums = df['joined'].astype(str).apply(nltk.word_tokenize)
avg_lexrank_lens = tokenized_sums.apply(len)

df = pd.read_csv('../results/lead_n_results.csv')
tokenized_sums = df['generated'].astype(str).apply(nltk.word_tokenize)
avg_lead_lens = tokenized_sums.apply(len)

df = pd.read_csv('../results/random_3.csv')
tokenized_sums = df['generated'].astype(str).apply(nltk.word_tokenize)
avg_r3_lens = tokenized_sums.apply(len)

fig, ax = plt.subplots()

sns.kdeplot(avg_r3_lens, ax=ax, label='Random 3')
sns.kdeplot(avg_lead_lens, ax=ax, label='Lead-n')
sns.kdeplot(avg_lexrank_lens, ax=ax, label='Lexrank')
fig.legend()

plt.savefig('avg_length_kde.png')
