import os
import time

import numpy as np
import pandas as pd 
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import pipeline, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available())
test_path = "../data/test.csv"
test = pd.read_csv(test_path)[-3:]

tokenizer = T5Tokenizer.from_pretrained('./summ_model')
model = T5ForConditionalGeneration.from_pretrained('./summ_model')

def clean(df):
    df['Text'] = df['Text'].apply(lambda x: x.strip().replace("\n", ""))
    df['Sum'] = df['Sum'].apply(lambda x: x.strip().replace("\n", ""))
    df['full_text'] = df['Text'].apply(lambda x: 'summarize: ' +x)
    return df 

test = clean(test)
print(test.shape)

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

test_dataset = SumDataset(tokenizer, test)
print(len(test_dataset))
print(test_dataset.__getitem__)

print('Dataset created')

model.to(device)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

triplets = []
model.eval()

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        print(f'Working on batch {i+1}:')
        input_ids = batch['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)
        print(len(input_ids))
        print(len(attention_mask))
        print(len(labels))
        generated = model.generate(input_ids = input_ids, attention_mask = attention_mask, max_length=150,
                num_beams=1, repetition_penalty=2.5, length_penalty=1.0, early_stopping=False
                )
        original = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in input_ids]
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
        triplets.extend(zip(original, pred, target))

print(triplets)
