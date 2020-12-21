import os

import sentence_transformers
import pandas as pd
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

r = 17
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
df = pd.read_csv('./data/data.csv', index_col=0) 

train, test = train_test_split(df, train_size=0.75, random_state=r)
train.to_csv('./train.csv')
test.to_csv('./test.csv')
train_texts = list(zip(train['question_1'].tolist(), train['question_2'].tolist()))
test_texts = list(zip(test['question_1'].tolist(), test['question_2'].tolist()))
train_labels = train['similar'].astype(float).tolist()
test_labels = test['similar'].astype(float).tolist()

ftr = dict(zip(train_texts, train_labels))
fte = dict(zip(test_texts, test_labels))

#texts=list(zip(df['question_1'].tolist(), df['question_2'].tolist()))
#labels=df['similar'].astype(float).tolist()
#
#oa = dict(zip(texts, labels))
#
#train, test = [i.to_dict() for i in train_test_split(pd.Series(oa), train_size=0.7, random_state=r)]

train_examples = []
test_examples = []

#for k, v in train.items():
#    train_examples.append(InputExample(texts=k, label=v))
#
#for k,v in test.items():
#    test_examples.append(InputExample(texts=k, label=v))

for k, v in ftr.items():
    train_examples.append(InputExample(texts=k, label=v))
for k, v in fte.items():
    test_examples.append(InputExample(texts=k, label=v))

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
model_save_path = './roberta-large-ft/'

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500, output_path=model_save_path)
