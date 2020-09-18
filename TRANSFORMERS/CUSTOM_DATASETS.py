## ================================== ##
##          CUSTOM DATASETS           ##
## ================================== ##

## 参考资料
#  https://huggingface.co/transformers/custom_datasets.html

## 编程环境
#  OS：Windows 10 Pro
#  Editor：VS Code
#  Python：3.7.7 (Anaconda 5.3.0)
#  PyTorch：1.5.1

__author__ = 'Atomicoo'


## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Sequence Classification with IMDb Reviews

# %%
from pathlib import Path
import requests
import tarfile

# %%
DATA_PATH = Path("data")
PATH = DATA_PATH / "aclImdb"

URL = "http://ai.stanford.edu/~amaas/data/sentiment/"
FILENAME = "aclImdb_v1.tar.gz"

if not (DATA_PATH / FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (DATA_PATH / FILENAME).open('wb').write(content)
else:
    print("## [INFO] data file already exists.".format(FILENAME))

if not PATH.exists():
    with tarfile.open((DATA_PATH / FILENAME), 'r') as tf:
        tf.extractall(DATA_PATH)
else:
    print("## [INFO] data directory already exists.")

# %%
def read_imdb_split(path):
    if isinstance(path, (str, Path)):
        if isinstance(path, str):
            path = Path(path)
    else:
        raise Exception("path format error")
    texts = []
    labels = []
    for lab_dir in ["pos", "neg"]:
        for fr in (path/lab_dir).iterdir():
            texts.append(fr.read_text(encoding='utf-8'))
            labels.append(0 if lab_dir is "neg" else 1)
    return texts, labels

# %%
train_texts, train_labels = read_imdb_split((PATH/"train"))
test_texts, test_labels = read_imdb_split((PATH/"test"))

# %%
from sklearn.model_selection import train_test_split

# %%
train_texts, val_texts, train_labels, val_labels = \
    train_test_split(train_texts, train_labels, test_size=.2)

# %%
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# %%
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# %%
train_encodings = tokenizer(train_texts, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, padding=True, truncation=True)
test_encodings = tokenizer(test_texts, padding=True, truncation=True)

# %%
import torch
from torch.utils.data import Dataset

# %%
class ImdbDataset(Dataset):
    """Imdb Reviews Dataset"""
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, index):
        item = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)

# %%
train_dataset = ImdbDataset(train_encodings, train_labels)
val_dataset = ImdbDataset(val_encodings, val_labels)
test_dataset = ImdbDataset(test_encodings, test_labels)

# %%
from transformers import Trainer, TrainingArguments

# %%
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# %%
trainer.train()
# trainer.evaluate()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

# %%
from torch.utils.data import DataLoader
from transformers import AdamW

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# %%
model.to(device)
model.train()

optim = AdamW(model.parameters(), lr=5e-5)

# %%
for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()



## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Token Classification with W-NUT Emerging Entities

# %%
from pathlib import Path
import requests
import zipfile
import re

# %%
DATA_PATH = Path("data")
PATH = DATA_PATH / "emerging_entities_17-master"

URL = "https://github.com/leondz/emerging_entities_17/archive/"
RAWNAME = "master.zip"
FILENAME = "emerging_entities_17-master.zip"

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
def read_wnut_17(path):
    if isinstance(path, (str, Path)):
        if isinstance(path, str):
            path = Path(path)
    else:
        raise Exception("path format error")
    raw_text = path.read_text(encoding='utf-8').strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs, tag_docs = [], []
    for doc in raw_docs:
        tokens, tags = [], []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tags)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs

# %%
tokens, tags = read_wnut_17((PATH/"wnut17train.conll"))

# %%
