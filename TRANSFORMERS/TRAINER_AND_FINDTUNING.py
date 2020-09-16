## ================================== ##
##       TRAINER AND FINDTUNING       ##
## ================================== ##

## 参考资料
#  https://huggingface.co/transformers/training.html

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


## @@@@@@@@@@@@@@@@@@@@@@
## Fine-tuning in native PyTorch

# %%
from transformers import BertForSequenceClassification
# 序列分类模型，初始化模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
model.train()

# 查看模型结构
for name, param in model.named_parameters():
    print(name, param.size())

# %%
from transformers import AdamW
# 可以使用任何PyTorch的优化器
# transformers提供了一些更高级封装的优化器，譬如：
# AdamW()：实现梯度偏差校正与权重衰减的Adam优化器
optimizer = AdamW(model.parameters(), lr=1e-5)
print(optimizer)

# %%
# transformers的optimizer允许（用户）为模型参数分组并应用不同的优化策略
# 譬如，bias与LayerNorm是不需要进行权重衰减的
no_decay = ['bias', 'LayerNorm']
optim_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optim_grouped_parameters, lr=1e-5)
print(optimizer)

# %%
from transformers import BertTokenizer
# 构建虚拟训练批次，了解（用户）训练/微调时需要传入模型的内容
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batched_text = ["I love Pixar.", "I don't care for Pixar."]
encoded_token = tokenizer(
                    batched_text, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt')
inputs, attentions = \
    encoded_token['input_ids'], encoded_token['attention_mask']
print("Inputs: \n{}\nAttens: \n{}".format(inputs, attentions))
labels = torch.tensor([1, 0])
print("Labels: {}".format(labels))

# %%
# 前向传播
# 返回交叉熵损失
outputs = model(inputs, attentions, labels=labels)
print("Loss: {}\nLogits: \n{}".format(outputs.loss, outputs.logits))

# %%
# 反向传播 + 更新权重
loss = outputs.loss
# 也可以自行计算交叉熵损失
# loss = F.cross_entropy(outputs.logits, labels)
loss.backward()
optimizer.step()

# %%
predictions = F.softmax(outputs.logits, dim=-1)
print(predictions)

# %%
from transformers import get_linear_schedule_with_warmup
# transformers还提供了一些learning rate scheduling tools
# 可以通过以下方式实现预热与学习率衰减
scheduler = get_linear_schedule_with_warmup(
                        optimizer=optimizer, 
                        num_warmup_steps=500, 
                        num_training_steps=5000)

# %%
# 要应用该scheduler，只需在权重更新后调用`scheduler.step()`
# optimizer.step(); scheduler.step()

# %%
# 冻结部分神经网络权重
# 与PyTorch相同，可以通过设置`.requires_grad=True`冻结部分神经网络权重
# 常见使用场景：（用户）只希望训练模型头部（即下游任务部分）的权重时
for param in model.base_model.parameters():
    param.requires_grad = False

# %%
# 学习率变化图
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer, 
                        num_warmup_steps=10, 
                        num_training_steps=50)
learning_rates = []
for _ in range(50):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    learning_rates.append(lr)

plt.plot(learning_rates)


## @@@@@@@@@@@@@@@@@@@@@@
## Fine-tuning with Trainer

# %%
from transformers import Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

#   - 使用 ~ trainer.train()：训练模型；trainer.evaluate()：评估模型
#   - model可以使用自定义Module，但请确保`.forward()`返回的第一个结果
#     为loss tensor
#   - Trainer会调用默认的内部方法来准备批次数据喂给模型，（用户）可以
#     自定义用于准备批次数据的方法，并传给`data_collator`参数

# %%
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# 自定义方法用于计算除loss外的其他评估指标，并传给`compute_metrics`参数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# %%
