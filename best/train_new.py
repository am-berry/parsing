import os
import time

import numpy as np
import pandas as pd
import torch
import pytorch-lightning as pl
from tqdm import tqdm

from torch.utils.data import dataset, dataloader
from transformers import t5tokenizer, t5forconditionalgeneration, trainer, trainingarguments
from sklearn.model_selection import train_test_split
from transformers import pipeline, adamw

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available())
tr_path = '../data/train.csv'
train = pd.read_csv(tr_path)
tr, val = train_test_split(train, test_size=0.2, random_state=17)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

def clean(df):
    df['Text'] = df['Text'].apply(lambda x: x.strip().replace("\n", ""))
    df['Sum'] = df['Sum'].apply(lambda x: x.strip().replace("\n", ""))
    df['full_text'] = df['Text'].apply(lambda x: 'summarize: ' +x)
    return df

tr = clean(tr)
val = clean(val)

class SumDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        source = self.tokenizer.batch_encode_plus(self.df.full_text, truncation=True, padding=True)
        target = self.tokenizer.batch_encode_plus(self.df.Sum.tolist(), truncation=True, padding=True)
        source.update(labels=target.get('input_ids'))
        source.update(decoder_attention_mask=target.get('attention_mask'))
        return {key: torch.tensor(val[idx]) for key, val in source.items()}

train_dataset = SumDataset(tokenizer, tr)
val_dataset = SumDataset(tokenizer, val)

print('Datasets created')

