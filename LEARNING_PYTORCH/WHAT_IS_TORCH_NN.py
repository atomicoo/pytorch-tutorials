## ================================== ##
##      WHAT IS TORCH.NN REALLY ?     ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/nn_tutorial.html
#  (AUTOGRAD MECHANICS) https://pytorch.org/docs/stable/notes/autograd.html

## 编程环境
#  OS：Windows 10 Pro
#  Editor：VS Code
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
## @@ MNIST data setup
## @@ Neural Net from scratch (no torch.nn)
## @@ Using torch.nn.functional
## @@ Refactor using nn.Module
## @@ Refactor using nn.Linear
## @@ Refactor using optim
## @@ Refactor using Dataset
## @@ Refactor using DataLoader
## @@ The whole final code
## @@ Switch to CNN
## @@ nn.Sequential
## @@ Wrapping DataLoader
## @@ Using your GPU
## 
## 

# torch.nn
#   - Module：包含状态（譬如神经网络权重）的可调用函数，大致对应神经网络中的Layers
#   - Parameter：Tensor的高一层封装，表示Module的参数
#   - functional：包括损失函数、激活函数以及各种神经网络Layers的无状态版本
# torch.optim
#   包括各种优化器，在反向传播时更新权重Parameter
# Dataset
#   具有`__len__`和`__getitem__`方法的抽象接口
# DataLoader
#   联合Dataset和Sampler，从Dataset中创建迭代器，每次返回一个批次的数据

## @@@@@@@@@@@@@@@@@@@
## MNIST data setup

# %%
# 使用pathlib包处理路径（赞）
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
# from IPython.core.debugger import set_trace

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
# PyTorch提供抽象的`Dataset`类。
# 可以通过子类化`torch.utils.data.Dataset`来快速构建自定义数据集
# 必须具有`__len__()`方法和`__getitem__()`方法
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
# valid_ds = TensorDataset(x_valid, y_valid)

inputs, targets = train_ds[bgn:end]


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Refactor using DataLoader

# %%
# `DataLoader`负责数据的载入，主要为批次的管理
# 联合了`Dataset`和`Sampler`，前者为数据来源，后者为提取策略
from torch.utils.data import DataLoader

train_dl = DataLoader(dataset=train_ds, batch_size=B, shuffle=True)
# valid_dl = DataLoader(dataset=valid_ds, batch_size=B*2)


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Add validation

# %%
# 需要验证集来确保训练过程不会过拟合
# 通过`.train()`和`.eval()`切换模式来确保某些特殊Layers在不同模式下操作的正确性
# 譬如，`nn.BatchNorm2D`或`nn.Dropout`
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(dataset=valid_ds, batch_size=B*2)

model.eval()
with torch.no_grad():
    valid_loss = sum(loss_fn(model(inputs), targets) 
                     for inputs, targets in valid_dl)


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## The whole final code

# %%
impoort math
import pickle
import gzip
import requests
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (PATH / FILENAME).open('wb').write(content)

with gzip.open((PATH / FILENAME), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


# 转换为torch.tensor格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensr, (x_train, y_train, x_valid, y_valid)
)

def get_data(train_ds, valid_ds, batch_size):
    return (
        DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(dataset=valid_ds, batch_size=batch_size*2),
    )

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

# 批次训练/评估
def run_batch(model, loss_fn, inputs, targets, opt=None):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    acc = accuracy(outputs, targets)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), acc.item(), len(inputs)

def fit(epochs, model, opt, train_dl, valid_dl):
    # 训练Loops
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_dl):
            loss, acc, num = run_batch(model, loss_fn, inputs, targets, opt)
            if (i+1) % 100 == 0:
                print("## Epoch {:2d}: Step {:5d}, Train Loss {:.4f}, Train Acc {:.2%}"
                    .format(epoch, i*B, loss, acc))
        model.eval()
        with torch.no_grad():
            losses, accs, nums = zip(
                *[run_batch(model, loss_fn, inputs, targets) for inputs, targets in valid_dl]
            )
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)
        print("## Epoch {:2d}: Valid Loss {:.4f}, Valid Acc {:.2%}"
            .format(epoch, valid_loss, valid_acc))

B = 64  # 批次尺寸
lr = 0.5  # 学习率
epochs = 2  # 迭代次数

# 创建数据集
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

# 数据载入
train_dl, valid_dl = get_data(train_ds, valid_ds, B)
# 获取模型
model, opt = get_model(Mnist_Logistic, lr)
# 训练模型
fit(epochs, model, opt, train_dl, valid_dl)

print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Switch to CNN

# 可以看到，在利用PyTorch中已封装好的包对原代码进行重构之后，训练代码非常简洁
# 同时训练代码与网络模型的解耦，使模型的切换变得非常方便

# %%
# 转换为卷积神经网络模型
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
    
    def forward(self, inputs):
        inputs = inputs.view(-1, 1, 28, 28)
        inputs = F.relu(self.conv1(inputs))
        inputs = F.relu(self.conv2(inputs))
        inputs = F.relu(self.conv3(inputs))
        inputs = F.avg_pool2d(inputs, 4)
        return inputs.view(-1, inputs.size(1))

lr = 0.1

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, opt, train_dl, valid_dl)

print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## nn.Sequential

# %%
from collections import OrderedDict

# 根据函数返回相应的神经网络Layers
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, inputs):
        return self.func(inputs)

# 利用`torch.nn.Sequencial`快速编写自定义神经网络
model = nn.Sequential(OrderedDict([
    ('iview', Lambda(lambda x: x.view(-1, 1, 28, 28))),
    ('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)),
    ('relu2', nn.ReLU()),
    ('conv3', nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)),
    ('relu3', nn.ReLU()),
    ('pool', nn.AvgPool2d(4)),
    ('oview', Lambda(lambda x: x.view(-1, x.size(1)))),
]))

lr = 0.1
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, opt, train_dl, valid_dl)

print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Wrapping DataLoader

# %%
preprocess = lambda x, y: (x.view(-1, 1, 28, 28), y)

# 封装DataLoader
class WrappingDataLoader:
    def __init__(self, dl, fn):
        self.dl = dl
        self.fn = fn
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.fn(*b)

train_dl, valid_dl = get_data(train_ds, valid_ds, B)
train_dl = WrappingDataLoader(train_dl, preprocess)
valid_dl = WrappingDataLoader(valid_dl, preprocess)

# 利用`torch.nn.Sequencial`快速编写自定义神经网络
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)),
    ('relu2', nn.ReLU()),
    ('conv3', nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)),
    ('relu3', nn.ReLU()),
    ('pool', nn.AdaptiveAvgPool2d(1)),
    ('oview', Lambda(lambda x: x.view(-1, x.size(1)))),
]))

lr = 0.1
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, opt, train_dl, valid_dl)

inputs, targets = inputs.view(-1, 1, 28, 28), targets
print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Using your GPU

# %%
print(torch.cuda.is_available())
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %%
preprocess = lambda x, y: (x.view(-1, 1, 28, 28).to(device), y.to(device))

train_dl, valid_dl = get_data(train_ds, valid_ds, B)
train_dl = WrappingDataLoader(train_dl, preprocess)
valid_dl = WrappingDataLoader(valid_dl, preprocess)

# 利用`torch.nn.Sequencial`快速编写自定义神经网络
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)),
    ('relu2', nn.ReLU()),
    ('conv3', nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)),
    ('relu3', nn.ReLU()),
    ('pool', nn.AdaptiveAvgPool2d(1)),
    ('oview', Lambda(lambda x: x.view(-1, x.size(1)))),
]))

lr = 0.1
model = model.to(device)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, opt, train_dl, valid_dl)

# %%
inputs, targets = inputs.view(-1, 1, 28, 28).to(device), targets.to(device)
print(loss_fn(model(inputs), targets), accuracy(model(inputs), targets))

# %%
