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

# Data loading into dataframe, setting GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available())
torch.backends.cudnn.benchmark = True
test_path = "../data/test.csv"
test = pd.read_csv(test_path)

# Model loading from trained dir 
tokenizer = T5Tokenizer.from_pretrained('./trained_model/')
model = T5ForConditionalGeneration.from_pretrained('./trained_model/')

# Arbitrary cleaning functions 
def clean(df):
    df['Text'] = df['Text'].apply(lambda x: str(x).strip().replace("\n", ""))
    df['Sum'] = df['Sum'].apply(lambda x: str(x).strip().replace("\n", ""))
    df['full_text'] = df['Text'].apply(lambda x: 'summarize: ' +x)
    return df 

test = clean(test)
print(test.shape)

# Subclass of Pytorch DataSet, performs tokenisation in batches 
class SumDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.df = df
        self.source = self.tokenizer.batch_encode_plus(self.df.full_text, truncation=True, padding=True)
        self.target = self.tokenizer.batch_encode_plus(self.df.Sum.tolist(), truncation=True, padding=True)
        self.source.update(labels=self.target.get('input_ids'))
        self.source.update(decoder_attention_mask=self.target.get('attention_mask'))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.source.items()}

test_dataset = SumDataset(tokenizer, test)

print('Dataset created')

# DataLoader with batch-size 16 
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
print('test loader done')
triplets = []

# Setting evaluate mode on model 
model.eval()
model.to(device)
print('model evaluate mode')

# Evaluation loop for beam search 
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        print(f'Working on batch {i+1}:')
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        generated = model.generate(input_ids = input_ids, attention_mask = attention_mask, max_length=75,
                num_beams=4, repetition_penalty=2.5, length_penalty=0.6, early_stopping=False
                )
        original = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in input_ids]
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
        triplets.extend(zip(original, pred, target))

print(len(triplets))
print('Beam 4 done')

# Outputting results 

final = pd.DataFrame(triplets, columns = ['original', 'pred_summary', 'orig_summary'])
final.to_csv('./finetuned_beam4.csv')
del final, triplets

# Evaluation loop for greedy search 
triplets = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        print(f'Working on batch {i+1}:')
        input_ids = batch['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)
        print(len(input_ids))
        print(len(attention_mask))
        print(len(labels))
        generated = model.generate(input_ids = input_ids, attention_mask = attention_mask, max_length=75)
        original = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in input_ids]
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
        triplets.extend(zip(original, pred, target))

print(len(triplets))
print('Greedy done')

# Outputting results 
final = pd.DataFrame(triplets, columns = ['original', 'pred_summary', 'orig_summary'])
final.to_csv('./finetuned_greedy.csv')
del final, triplets

# Evaluation loop for nucleus sampling 
triplets = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        print(f'Working on batch {i+1}:')
        input_ids = batch['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)
        generated = model.generate(input_ids = input_ids, attention_mask = attention_mask, do_sample=True, max_length=75,
                top_p=0.95, top_k=50)
        original = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in input_ids]
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
        triplets.extend(zip(original, pred, target))

print(len(triplets))
print('Nucleus done')

# Outputting results 
final = pd.DataFrame(triplets, columns = ['original', 'pred_summary', 'orig_summary'])
final.to_csv('./finetuned_nucleus.csv')

# Rouge calculations are computed in rouge_calcs.py
