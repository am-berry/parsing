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

print('Datasets created')

"""
training_args = TrainingArguments(
        output_dir='./results/',
        num_train_epochs=5,
        per_device_train_batch_size=64, 
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
"""

model.to(device)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

print('Beginning training now: ')
train_losses = []
val_losses = []
for epoch in tqdm(range(25)):
    running_loss = 0.0
    running_val_loss = 0.0
    for batch in train_loader:
        model.train()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss=outputs[0]
        running_loss += loss * len(input_ids)  
        loss.backward()
        optim.step()
    with torch.no_grad():
        model.eval()
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

train_losses = np.insert(np.array(train_losses), 0, 0)
val_losses = np.insert(np.array(val_losses), 0, 0)

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(left=1)
plt.savefig('losses.png')
model.save_pretrained("./summ_model/summ_model")
tokenizer.save_pretrained("./summ_model/summ_model")
