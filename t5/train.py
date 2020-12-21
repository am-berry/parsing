import os
import time

import numpy as np
import pandas as pd 
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import pipeline, AdamW

# Loading in, setting GPU 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available())
tr_path = '../data/train.csv'
train = pd.read_csv(tr_path)
tr, val = train_test_split(train, test_size=0.2, random_state=17)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Arbitrary cleaning functions
def clean(df):
    df['Text'] = df['Text'].apply(lambda x: x.strip().replace("\n", ""))
    df['Sum'] = df['Sum'].apply(lambda x: x.strip().replace("\n", ""))
    df['full_text'] = df['Text'].apply(lambda x: 'summarize: ' +x)
    return df 

tr = clean(tr)
val = clean(val)

# Creating DataSet subclass for our data
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
        return {key: torch.tensor(val[idx],dtype=torch.long).squeeze() for key, val in self.source.items()}

train_dataset = SumDataset(tokenizer, tr)
val_dataset = SumDataset(tokenizer, val)

print('Datasets created')

# Creating DataLoader and optimiser
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

print('Beginning training now: ')
train_losses = []
val_losses = []

# Training loop 
for epoch in tqdm(range(3)):
    running_loss = 0.0
    running_val_loss = 0.0
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_mask = batch['decoder_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_mask, labels=labels)
        loss = outputs[0]
        running_loss += loss * len(input_ids)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % 5000 == 0:
            model.save_pretrained(f'./models/modelchkpt_epoch{epoch}_step{i}/') 
            tokenizer.save_pretrained(f'./models/modelchkpt_epoch{epoch}_step{i}/') 
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            running_val_loss += outputs[0] * len(input_ids)
    rl = (running_loss/len(train_dataset)).cpu().detach().numpy()
    vl = (running_val_loss/len(val_dataset)).cpu().detach().numpy()
    train_losses.append(rl)
    val_losses.append(vl)
    print(f'Epoch {int(epoch)+1}: train loss {rl}, val loss {vl}')
    model.save_pretrained(f'./models/modelchkpt{epoch}/')
    tokenizer.save_pretrained(f'./models/modelchkpt{epoch}/')

# Saving model 
save_out = "./trained_model/"
model.save_pretrained(save_out)
tokenizer.save_pretrained(save_out)
print(train_losses)
print(val_losses)

# Plotting graphs of epoch losses 
train_losses = np.insert(np.array(train_losses), 0, 0)
val_losses = np.insert(np.array(val_losses), 0, 0)

with open('train_results.json', 'w') as f:
    json.dump(train_losses.tolist(), f)
with open('val_results.json', 'w') as f:
    json.dump(val_losses.tolist(), f)
    
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(left=1)
plt.savefig('losses.png')
