import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/test.csv')

df['len_text_chars'] = df['Text'].apply(len)
df['tokenized_text'] = df['Text'].astype(str).apply(nltk.sent_tokenize)
df['len_text_sentences'] = df['tokenized_text'].apply(len)
df['tokenized_words'] = df['Text'].astype(str).apply(nltk.word_tokenize)
df['len_text_words'] = df['tokenized_words'].apply(len)
df.dropna(inplace=True)

fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10,5))
fig.suptitle('Bigger 1 row x 2 columns axes with no data')
axes[0].set_title('Source text length in characters')
axes[1].set_title('Source text length in words')
axes[2].set_title('Source text length in sentences')

sns.histplot(ax=axes[0], data=df['len_text_chars'])
sns.histplot(ax=axes[1], data=df['len_text_words'])
sns.histplot(ax=axes[2], data=df['len_text_sentences'])
plt.savefig('asdf.png')
