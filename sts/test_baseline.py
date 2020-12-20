import sentence_transformers
import pandas as pd
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

test = pd.read_csv('./test.csv', index_col=0) 

test_texts = list(zip(test['question_1'].tolist(), test['question_2'].tolist()))
test_labels = test['similar'].astype(float).tolist()

fte = dict(zip(test_texts, test_labels))

test_examples = []

for k, v in fte.items():
    test_examples.append(InputExample(texts=k, label=v))

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
model.evaluate(evaluator, output_path='./results/')
