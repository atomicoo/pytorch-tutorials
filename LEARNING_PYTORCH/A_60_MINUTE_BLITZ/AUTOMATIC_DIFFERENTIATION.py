## ================================== ##
## DL WITH PYTORCH: A 60 MINUTE BLITZ ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
#  https://atomicoo.com/technology/pytorch-auto-diff-autograd/

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


## @@@@@@@@@@@
## 自动求导

# AutoGrad包是PyTorch中所有神经网络的核心，
# 为张量上的所有操作提供自动求导。
# 它是一个运行时定义的框架，
# 即反向传播是随着对张量的操作来逐步决定的，
# 这也意味着在每个迭代中都可以是不同的。

##########
## 张量
#  torch.Tensor类的重要属性/方法：
#    - dtype:   该张量存储的值类型，可选类型见：torch.``dtype；
#    - device:  该张量存放的设备类型，cpu/gpu
#    - data:    该张量节点存储的值；
#    - requires_grad: 表示autograd时是否需要计算此tensor的梯度，默认False；
#    - grad:    存储梯度的值，初始为None；
#    - grad_fn: 反向传播时，用来计算梯度的函数；
#    - is_leaf: 该张量节点在计算图中是否为叶子节点；

# %%
# 将`.requires_grad`设置为true，追踪Tensor上的所有操作
x = torch.ones(2, 2, requires_grad=True)
print(x)

# %%
# `.grad_fn`属性指向Function，编码Tensor间的运算，
# 包含`.forward()`和`.backward()`
y = x + 2
print(y)
print(y.grad_fn)

# %%
z = y * y * 3
out = z.mean()
print(z, out)

# %%
# 就地改变`.requires_grad`属性
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# %%
# 通过调用`.detach()`阻止追踪Tensor操作（历史记录）
# 或通过将代码块包裹在`with torch.no_grad():`中（评估模型时常用）
a.detach().fill_(9.)
print(a)
# or
with torch.no_grad():
    a[:] = 11.
print(a)


##########
## 梯度
# %%
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

# %%
# 调用`torch.backward()`自动反向传播计算梯度，
# 并将梯度累积到`.grad`属性中
# 注意，只有`.requires_grad`与`.is_leaf`同时为True的Tensor才会累积梯度
out.backward(create_graph=True)
# 此处`out.backward()`等价于`out.backward(torch.tensor(1.))`
# 因为out为1x1张量，所以`.backward()`参数可以省略
print(x.grad)

# %%
# 自动计算梯度还可以使用以下方法调用
torch.autograd.backward(out, create_graph=True)
print(x.grad)
# or
torch.autograd.backward(out, create_graph=True)
print(x.grad)

# %%
# 关于`.backward()`为什么需要参数`grad_tensor`的存在，
# 见下，具体参考 https://zhuanlan.zhihu.com/p/83172023

# %%
x = torch.ones(2,requires_grad=True)
z = x + 2
z.backward(torch.ones_like(z))
print(x.grad)

# %%
# 由于反向传播中梯度是进行累积的，所以当输出out为矩阵时
# 直接反向计算梯度与先将out元素求和再反向计算梯度的结果是一样的
# 通过设置`grad_tensor`参数可以“加权”求和
x = torch.ones(2,requires_grad=True)
z = x + 2
z.sum().backward()
print(x.grad)

# %%
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

# %%
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# %%
# CLASS torch.autograd.Function
# https://pytorch.org/docs/stable/autograd.html#function

# %%
