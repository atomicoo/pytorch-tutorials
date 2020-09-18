## ================================== ##
##        SUMMARY OF THE TASKS        ##
## ================================== ##

## 参考资料
#  https://huggingface.co/transformers/task_summary.html

## 编程环境
#  OS：Windows 10 Pro
#  Editor：VS Code
#  Python：3.7.7 (Anaconda 5.3.0)
#  PyTorch：1.5.1

__author__ = 'Atomicoo'


## @@@@@@@@@@@@@@@@@@@@@@@
## Sequence Classification

# %%
from transformers import pipeline

nlp = pipeline('sentiment-analysis')

sequences = [
    "We are very happy to show you the Transformers library.", 
    "We hope you don't hate it."]

results = nlp(sequences)

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classes = ['NEGATIVE', 'POSITIVE']

sequences = [
    "We are very happy to show you the Transformers library.", 
    "We hope you don't hate it."]
inputs = tokenizer(sequences, 
                   padding=True, 
                   return_tensors='pt')

outputs = model(**inputs)
values, indices = torch.softmax(outputs[0], dim=1).max(dim=1)
results = [{'label': classes[i], 'score': p} \
    for i, p in zip(indices.tolist(), values.tolist())]

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Extractive Question Answering

# %%
from transformers import pipeline

nlp = pipeline("question-answering")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
"""
questions = [
    "What is extractive question answering?",
    "What is a good example of a question answering dataset?"]

for question in questions:
    result = nlp(question=question, context=context)
    print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

# %%
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = 'distilbert-base-cased-distilled-squad'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
"""
questions = [
    "What is extractive question answering?",
    "What is a good example of a question answering dataset?"]

for question in questions:
    inputs = tokenizer(question, 
                       context, 
                       add_special_tokens=True, 
                       return_tensors='pt')
    input_ids = inputs['input_ids'][0]
    ans_start_scores, ans_end_scores = model(**inputs)

    ans_start = torch.argmax(ans_start_scores)
    ans_end = torch.argmax(ans_end_scores) + 1
    ans_tokens = tokenizer.convert_ids_to_tokens(input_ids[ans_start:ans_end])
    ans_string = tokenizer.convert_tokens_to_string(ans_tokens)
    result = {'answer': ans_string, 'start': ans_start, 'end': ans_end}
    print(f"Answer: '{result['answer']}', start: {result['start']}, end: {result['end']}")


## @@@@@@@@@@@@@@@@@
## Language Modeling

###########################
## Masked Language Modeling

# %%
from transformers import pipeline

nlp = pipeline("fill-mask")

sequences = [
    f"HuggingFace is creating a {tokenizer.mask_token} that the community uses to solve NLP tasks.",
    f"Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."]

result = nlp(sequences)

from pprint import pprint
pprint(result)

# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

sequences = [
    f"HuggingFace is creating a {tokenizer.mask_token} that the community uses to solve NLP tasks.",
    f"Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."]

inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
input_ids = inputs['input_ids']
mask_token_indexes = torch.where(input_ids==tokenizer.mask_token_id)[1].tolist()

outputs = model(**inputs)[0]
mask_token_logits = torch.stack([outputs[i, mask_token_indexes[i], :] \
                                for i in range(len(outputs))])
mask_token_logits = torch.softmax(mask_token_logits, dim=-1)

top_5_scores, top_5_tokens = torch.topk(mask_token_logits, k=5, dim=-1)
top_5_scores, top_5_tokens = top_5_scores.tolist(), top_5_tokens.tolist()

results = []
for index in range(len(top_5_tokens)):
    tokens, scores = top_5_tokens[index], top_5_scores[index]
    result = []
    for token, score in zip(tokens, scores):
        seq = sequences[index].replace(tokenizer.mask_token, tokenizer.decode([token]))
        result.append({'score': score, 'sequence': seq, 'token': token, 'token_str': tokenizer.convert_ids_to_tokens(token)})
    results.append(result)

from pprint import pprint
pprint(results)


## @@@@@@@@@@@@@@@@@@@@@@@@
## Named Entity Recognition

# %%
from transformers import pipeline

nlp = pipeline("ner")

sequences = [
    "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO", 
    "therefore very close to the Manhattan Bridge."]
result = nlp(sequences)

from pprint import pprint
pprint(result)

# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

sequences = [
    "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO", 
    "therefore very close to the Manhattan Bridge."]

tokens = tokenizer.convert_ids_to_tokens(tokenizer(sequences[0])['input_ids'])
inputs = tokenizer(sequences, 
                   padding=True,
                   truncation=True,
                   return_tensors='pt')
input_ids = inputs['input_ids']

outputs = model(**inputs)[0]
token_logits = torch.softmax(outputs, dim=-1)

label_scores, label_indices = torch.max(token_logits, dim=-1)
label_scores, label_indices = label_scores.tolist(), label_indices.tolist()

# %%
results = []
for i, (indices, scores) in enumerate(zip(label_indices, label_scores)):
    result = []
    for index, (indice, score) in enumerate(zip(indices, scores)):
        if indice == 0: continue
        result.append({
            'entity': label_list[indice], 
            'index': index, 
            'score': score, 
            'word': tokenizer.convert_ids_to_tokens([input_ids[i, index]])[0]
        })
    results.append(result)

from pprint import pprint
pprint(results)

# %%
