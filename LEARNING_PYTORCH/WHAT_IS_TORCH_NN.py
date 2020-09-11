## ================================== ##
##      WHAT IS TORCH.NN REALLY ?     ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/nn_tutorial.html
#  (AUTOGRAD MECHANICS) https://pytorch.org/docs/stable/notes/autograd.html

## 编程环境
#  OS：Win10 专业版
#  IDE：VS Code
#  Python：3.7.7 (Anaconda 5.3.0)
#  PyTorch：1.5.1

__author__ = 'Atomicoo'

# %%
from __future__ import print_function

import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


########################################
## C O N T E N T S
## 
## @@ torch.nn
## @@ torch.optim
## @@ Dataset
## @@ DataLoader
## 
## 

## @@@@@@@@@@@@@@@@@@@
## MNIST data setup

# %%
# 使用pathlib包处理路径
from pathlib import Path
# 使用requests下载数据
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (PATH / FILENAME).open('wb').write(content)

# %%
# 数据存储使用pickle序列化
import pickle
# 数据文件使用gzip格式压缩
import gzip

with gzip.open((PATH / FILENAME), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# print(x_train.shape)

# %%
import matplotlib.pyplot as plt
# 数据保存为为numpy.ndarray格式
import numpy as np

plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')

# %%
# 转换为torch.tensor格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
N, C = x_train.shape

print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Neural Net from scratch (no torch.nn)

# %%
import math

# 定义可学习参数
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
print(weights.shape)
bias = torch.zeros(10)
bias.requires_grad_()
print(bias.shape)

# %%
# 定义模型
def log_softmax(x):
    # `.unsqueeze(dim)`：添加一个size=1的轴作为张量的第dim个轴
    # `.squeeze(dim)`：去除张量中size=1的轴，若指定dim则先判断第dim个轴size是否为1
    return x - x.exp().sum(-1).log().unsqueeze(-1)
    # return (x.t()-x.exp().sum(-1).log()).t()
    # 广播机制需要最后一轴的尺寸匹配

def model(x):
    return log_softmax(x @ weights + bias)

# %%
# batch_size
B = 64

# %%
# 小批量数据
inputs = x_train[0:B]
outputs = model(inputs)
print(outputs[0], outputs.shape)

# %%
# 定义损失函数
def nll(outputs, targets):
    # NLLLoss()损失函数：
    # softmax()得到概率值在[0,1]，log()后的值在[-inf,0]，
    # log_softmax()的值（为负数）距零越远，对应的可能性越小
    # NLLLoss()损失会根据targets拿到对应的log_softmax()值，取相反数后再求平均值mean()
    return -outputs[range(targets.shape[0]), targets].mean()
loss_fn = nll

# %%
targets = y_train[0:B]
print(loss_fn(outputs, targets))

# %%
# 计算准确率
def accuracy(outputs, targets):
    predictions = torch.argmax(outputs, dim=-1)
    return (predictions == targets).float().mean()

# %%
print(accuracy(outputs, targets))

# %%
# 训练Loops
from IPython.core.debugger import set_trace

lr = 0.5  # 学习率
epochs = 2  # 迭代次数
for epoch in range(epochs):
    for i in range((N-1)//B+1):
        # set_trace()
        bgn = i * B
        end = bgn + B
        inputs = x_train[bgn:end]
        targets = y_train[bgn:end]
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
        if (i+1) % 100 == 0:
            print("## Epoch {:2d}, Step {:5d}, Loss {:.4f}, Acc {:.2%}"
                .format(epoch, end, loss, acc))

# %%
print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Using torch.nn.functional

# %%
# cross_entropy = log_softmax + nll_loss
# 交叉熵损失 = Log-Softmax + 负对数似然损失
loss_fn = F.cross_entropy

def model(x):
    return x @ weights + bias


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Refactor using nn.Module

# %%
# 通过子类化nn.Module自定义模型类
# 可学习参数在初始化函数中通过nn.Parameter来定义
# `.forward()`方法在子类中必须重写
# PyTorch会自动调用`.forward()`方法进行前向传播
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))
    
    def forward(self, inputs):
        return inputs @ weights + bias

model = Mnist_Logistic()

# %%
with torch.no_grad():
    # 通过`.parameters()`遍历模型参数并更新
    # 通过`.zero_grad()`对参数手动梯度清零
    for param in model.parameters():
        param -= param.grad * lr
    model.zero_grad()


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Refactor using nn.Linear

# %%
# torch.nn封装了一组常用的Modules，对应神经网络Layers，方便构建模型
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
    
    def forward(self, inputs):
        return self.lin(inputs)


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Refactor using optim

# %%
# 定义优化器
opt = optim.Adam(model.parameters(), lr=lr)
# 更新参数权重
opt.step()
# 参数梯度清零
opt.zero_grad()


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Refactor using Dataset

# %%
from torch.utils.data import TensorDataset


# %%
import math

# 定义模型
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
    
    def forward(self, inputs):
        return self.lin(inputs)

def get_model(model, lr):
    net = model()
    opt = optim.Adam(net.parameters(), lr=lr)
    return net, opt

# 定义损失函数：交叉熵损失
loss_fn = F.cross_entropy

# 计算准确率
def accuracy(outputs, targets):
    predictions = torch.argmax(outputs, dim=-1)
    return (predictions == targets).float().mean()

# 训练Loops
B = 64  # 批次尺寸
lr = 1e-3  # 学习率
epochs = 2  # 迭代次数

# 获取模型
model, opt = get_model(Mnist_Logistic, lr)

# 训练Loops
for epoch in range(epochs):
    for i in range((N-1)//B+1):
        bgn = i * B
        end = bgn + B
        inputs = x_train[bgn:end]
        targets = y_train[bgn:end]
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (i+1) % 100 == 0:
            print("## Epoch {:2d}, Step {:5d}, Loss {:.4f}, Acc {:.2%}"
                .format(epoch, end, loss, acc))

# %%
print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))

# %%
