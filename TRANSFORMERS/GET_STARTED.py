## ================================== ##
##         AUTOGRAD MECHANICS         ##
## ================================== ##

## 参考资料
#  https://huggingface.co/transformers/quicktour.html

## 编程环境
#  OS：Windows 10 Pro
#  Editor：VS Code
#  Python：3.7.7 (Anaconda 5.3.0)
#  PyTorch：1.5.1

__author__ = 'Atomicoo'

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


## @@@@@@@@@@@@@@
## Quick tour

################################
## Getting started with pipeline
###
# 在给定任务上使用预训练模型最简单粗暴的方式是`pipeline()`
#   - Sentiment analysis
#   - Text generation (in English)
#   - Name entity recognition (NER)
#   - Name entity recognition (NER)
#   - Filling masked text
#   - Summarization
#   - Translation
#   - Feature extraction
# %%
from transformers import pipeline

# %%
# Pipeline
classifier = pipeline('sentiment-analysis')

# %%
results = classifier([
    "We are very happy to show you the Transformers library.", 
    "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# %%
# 模型中心：https://huggingface.co/models
# 可以根据需要在模型中心查询相应的模型
# 譬如：
# classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# %%
# 任务汇总：https://huggingface.co/transformers/task_summary.html
# 查询常见NLP任务对应的模型及其使用方式

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# %%
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# %%
# 微调模型示例：https://huggingface.co/transformers/examples.html


################################
## Under the hood: pretrained models
###
# 使用`.from_pretrained()`创建`model`和`tokenizer`
# %%
# 创建model和tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# 使用tokenizer
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(inputs)

# %%
# 使用tokenizer（更多参数的使用）
inputs = tokenizer(
            [ "We are very happy to show you the Transformers library.",
              "We hope you don't hate it." ],
            padding=True,       # 是否填充，相同长度
            truncation=True,    # 是否截断，相同长度
            return_tensors="pt")
for key, value in inputs.items():
    print(f"{key}: {value.numpy().tolist()}")

# %%
# 模型是标准的`torch.nn.Module`模型（若使用TensorFlow则是`tf.keras.Model`模型）
# 可以使用PyTorch的方式查看它的所有参数
for param in model.named_parameters():
    print(param[0], param[1].size())

# %%
# 使用model
outputs = model(**inputs)
print(outputs)

# %%
# 这里我们得到一个tuple，还需要应用softmax来获得最终预测
predictions = F.softmax(outputs[0], dim=-1)
print(predictions)

# %%
# Transformers还提供一个`Trainer`类用于训练（微调）模型
# 详见：https://huggingface.co/transformers/training.html

# %%
# 微调模型后，通过以下方式保存（连同tokenizer）
SAVE_DIR = Path("pretrained/distilbert-base-uncased-finetuned-sst-2-english")
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

# %%
# 然后可以在需要时通过`.from_pretrained()`重新加载
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModel.from_pretrained(SAVE_DIR)

# 在TensorFlow中加载PyTorch模型：
# model = TFAutoModel.from_pretrained(SAVE_DIR, from_pt=True)
# 在PyTorch中加载TensorFlow模型：
# model = AutoModel.from_pretrained(SAVE_DIR, from_tf=True)

# %%
# 还可以令模型返回所有隐藏状态和注意力权重
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
hidden_states, attentions = outputs[-2:]

# %%
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 根据已知模型架构选择对应的模型类进行实例化，与上述使用`AutoModel`效果一样
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# %%
# 自定义模型
# 每种模型架构都有相应的配置类（譬如，Customizing the model -> DistilBertConfig）
# 可以通过修改配置参数（包括hidden dimension, dropout rate等）改变模型，

from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification

# %%
# 若进行核心修改（譬如hidden dimension），则无法使用预训练模型，需要从头开始训练
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)

# %%
# 若仅修改模型的头部（下游任务部分，譬如修改分类标签数），则仍然可以使用预训练模型
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# %%
extractor = pipeline("feature-extraction")

# %%
features = extractor(
    [ "We are very happy to show you the Transformers library.",
      "We hope you don't hate it." ])
print(torch.tensor(features).size())


## @@@@@@@@@@@@@@
## Philosophy

################################
## Main concepts
###
# 每种模型围绕3个类构建：
#   - Model 模型类：各种模型，可以使用预训练权重或从头训练权重
#   - Config 配置类：Model类的一部分，在不改变模型结构的情况下模型将自动实例化配置
#   - Tokenizer 类：存储模型词汇表，提供编/解码输入的方法
# 在上述3个类基础上，还提供了2个API：
#   - pipeline() 用于快速地使用模型
#   - Trainer() 用于方便地训练/微调模型
# 保存/加载模型：
#   - from_pretrained()
#   - save_pretrained()

# %%
