import os
import time

import numpy as np
import pandas as pd 
import torch

from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import pipeline

tr_path = '../data/train.csv'
train = pd.read_csv(tr_path)
tr, val = train_test_split(train, test_size=0.2, random_state=17)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

def clean(df):
    df['Text'] = df['Text'].apply(lambda x: x.strip().replace("\n", ""))
    df['Sum'] = df['Sum'].apply(lambda x: x.strip().replace("\n", ""))
    return df 

class SumDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.df = clean(df)
        self.full_text = self.df['Text'].apply(lambda x: 'summarize: ' + x) 
        self.text = self.tokenizer.batch_encode_plus(self.full_text, truncation=True, padding=True)
        self.summaries = self.tokenizer.batch_encode_plus(self.df.Sum, truncation=True, padding=True)
        self.text.update(labels=self.summaries.get('input_ids'))
        self.text.update(decoder_attention_mask=self.summaries.get('attention_mask'))

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.text.items()}

train_dataset = SumDataset(tokenizer, tr)
val_dataset = SumDataset(tokenizer, val)

training_args = TrainingArguments(
        output_dir='./results/',
        num_train_epochs=5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs/',
        )

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
        )

trainer.train()
trainer.save_model("./summ_model")
trainer.evaluate()
