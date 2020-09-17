# %%
from pathlib import Path
import requests
import zipfile
import numpy as np
import pandas as pd

# %%
DATA_PATH = Path("data")
PATH = DATA_PATH / "cola_public"

URL = "https://nyu-mll.github.io/CoLA/"
FILENAME = "cola_public_1.1.zip"

if not (DATA_PATH / FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (DATA_PATH / FILENAME).open('wb').write(content)
else:
    print("## [INFO] data zip file already exists.".format(FILENAME))

if not PATH.exists():
    with zipfile.ZipFile((DATA_PATH / FILENAME), 'r') as zf:
        zf.extractall(DATA_PATH)
else:
    print("## [INFO] data directory already exists.")

# %%
RAW_PATH = PATH / 'raw'
TRAIN_FILE = RAW_PATH / 'in_domain_train.tsv'
DEV_FILE = RAW_PATH / 'in_domain_dev.tsv'
OUT_DEV_FILE = RAW_PATH / 'out_of_domain_dev.tsv'

# %%
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# %%
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# %%
import torch
from torch.utils.data import Dataset, DataLoader

# %%
class ColaDataset(Dataset):
    """COLA Dataset"""
    def __init__(self, tsv_file, tokenizer):
        super().__init__()
        with open(tsv_file, 'r', encoding='utf-8') as fr:
            lines = [line.strip().split('\t') for line in fr.readlines()]
        self.labels = [int(line[1]) for line in lines]
        self.texts = [line[3] for line in lines]
        self.tokenizer = tokenizer
        self.encoded_tokens = tokenizer(self.texts,
                                        padding=True,
                                        truncation=True, 
                                        return_tensors='pt')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encoded_tokens.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

# %%
train_ds = ColaDataset(tsv_file=TRAIN_FILE, tokenizer=tokenizer)
dev_ds = ColaDataset(tsv_file=DEV_FILE, tokenizer=tokenizer)

# %%
# train_dl = DataLoader(dataset=train_ds, batch_size=4, shuffle=True, num_workers=0)
# dev_dl = DataLoader(dataset=dev_ds, batch_size=4, shuffle=False, num_workers=0)
# print(iter(train_dl).next())

# %%
from transformers.trainer import Trainer, TrainingArguments

# %%
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    tokenizer=tokenizer
)

# %%
trainer.train()

# %%
trainer.evaluate()

# %%
# Weights & Biases
import wandb
wandb.login()

# %%
