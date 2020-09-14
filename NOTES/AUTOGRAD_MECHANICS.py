## ================================== ##
##         AUTOGRAD MECHANICS         ##
## ================================== ##

## 参考资料
#  https://pytorch.org/docs/stable/notes/autograd.html
#  https://pytorch.apachecn.org/docs/1.4/56.html

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


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Excluding subgraphs from backward

#############################
## requires_grad

# %%
# 通过设置`.requires_grad`属性
# 根据需要从后向传播过程中排除某些子图
x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn(5, 5, requires_grad=True)
a = x + y
print(a.requires_grad)
b = a + z
print(b.requires_grad)

# %%
import torchvision

model = torchvision.models.resnet18(pretrained=True)
for param in model.named_parameters():
    print(param[0], param[1].size(), param[1].requires_grad)

# %%
# 冻结与训练网络模型
for param in model.parameters():
    param.requires_grad_(False)
# 替换最后一层网络，使用torch.nn构建Layers时
# 参数nn.Parameter的`.requires_grad`属性默认为True
# 由原本的1000分类重新训练为100分类模型
model.fc = nn.Linear(512, 100)
# for param in model.named_parameters():
#     print(param[0], param[1].size(), param[1].requires_grad)
# 仅更新最后一层的权重
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## How autograd encodes the history

# %%
# PyTorch的自动求导Autograd包是其所有神经网络的核心。
# Autograd会记录追踪（用户）构建模型时对tensors的所有操作，形成一个前向传播的有向无环图DAG，
# 称为计算图，输入tensors作为叶节点，输出tensor作为根节点，反向传播时从根节点开始遍历计算图
# 并根据链式法则层层更新梯度，直到所有`.requires_grad`为True的tensors梯度都得到更新

# 有图有真相：
# ![computation_graphs.gif](https://i.loli.net/2020/09/13/TH8kLbfUVAslR1M.gif)

# 需要注意的是，每次迭代都会重新构建模型，也即（用户）可以随时方便地改变模型的结构

inputs = torch.randn(4, 3, 64, 64)
outputs = model(inputs)

# 尝试追溯反向传播过程
print(outputs.grad_fn)
print(outputs.grad_fn.next_functions[0][0])
print(outputs.grad_fn.next_functions[2][0])
print(outputs.grad_fn.next_functions[2][0].next_functions[0][0])


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## In-place operations with autograd

#############################
## In-place correctness checks
# %%
# PyTorch非常非常不提倡使用in-place操作，不当的in-place操作将影响求导的正确性，
# 因此在实践中能不使用就尽量不使用。
# 很人性化的一点是，PyTorch会在（用户）使用不当的in-place操作时以报错的方式提醒，
# 因此在（用户）使用了in-place操作但程序未报错的情况下，可以确保梯度计算的正确性

# in-place操作：在不改变内存地址情况下修改数据
x = torch.randn(4, 3)
print(id(x))
print(x._version)
# NO
x = x.exp()
print(id(x))        # 内存地址不变
print(x._version)
# YES
x[1, 2] = 11
print(id(x))        # 内存地址改变
print(x._version)   # 通过`._version`来判断是否发生in-place操作`
                    # 每次in-place操作，`._version`就+1


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## Multithreaded Autograd

# %%
import threading

# 定义将要用于多线程的训练函数（simple example）
def train(index=i):
    x = torch.randn(5, 5, requires_grad=True)
    # forward pass
    y = (x + 3) * (x + 4) * 0.5
    # backward pass
    y.sum().backward()
    # potential optimizer update
    print(index)

# 多线程并发运行训练代码
threads = []
for i in range(10):
    p = threading.Thread(target=train, args=(i,))
    p.start()
    threads.append(p)
for p in threads:
    p.join()


#############################
## Concurrency on CPU

#############################
## Non-determinism

#############################
## Graph retaining

# 当计算图的一部分单线程运行，而另一部分在多线程中并行，
# 则计算图的第一部分将被共享，这种情况下某线程可能会
# 破坏计算图（因为PyTorch采用动态图）而导致其他线程崩溃
# 此时可以参考使用`retain_graph=True`参数


#############################
## Thread Safety on Autograd Node

#############################
## Autograd for Complex Numbers
# %%
