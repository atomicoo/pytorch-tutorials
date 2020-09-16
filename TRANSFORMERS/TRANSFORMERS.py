# %%
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

# %%
MODEL_NAME = 'bert-base-cased'
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
tokens = tokenizer.tokenize("this is an input example")
print("Tokens: {}".format(tokens))

tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))

tokens_pt = torch.tensor([tokens_ids])
print("Tokens PyTorch: {}".format(tokens_pt))

# %%
outputs, pooled = model(tokens_pt)
print("Token wise output: {}\nPooled output: {}".format(outputs.shape, pooled.shape))

# BERT outputs two tensors:
#   - One with the generated representation for every token in the input (1, NB_TOKENS, REPRESENTATION_SIZE)
#   - One with an aggregated representation for the whole input (1, REPRESENTATION_SIZE)

# %%
# 
tokens = tokenizer("this is an input example", return_tensors='pt')
for key, val in tokens.items():
    print("{}:\n\t{}".format(key, val))

# %%
