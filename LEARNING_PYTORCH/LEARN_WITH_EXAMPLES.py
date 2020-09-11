## ================================== ##
## DL WITH PYTORCH: A 60 MINUTE BLITZ ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

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

import numpy as np
from collections import OrderedDict
import random

########################################
## C O N T E N T S
## 
## @@ Tensors
## @@ AutoGrad
## @@ nn.Module
## 
##

## @@@@@@@@@@@
## Tensors

####################
## Warm-up: NumPy
# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# %%
# create random inputs and outputs
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# %%
# randomly init weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# %%
learning_rate = 1e-6
for t in range(500):
    # forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    # print loss
    loss = np.square(y_pred-y).sum()
    print(t, loss)
    # backprop
    grad_y_pred = 2.*(y_pred-y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)
    # update weight
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2


#####################
## PyTorch: Tensor

##`ndarray` -> `tensor`
# `ndarray.dot()` -> `tensor.mm()`
# `ndarray.maximum()` -> `tensor.clamp()`
# `ndarray.T` -> `tensor.t()`
# `ndarray.square()` -> `tensor.pow(2)`
# `ndarray.copy()` -> `tensor.clone()`

# %%
dtype = torch.float
device = torch.device('cpu')
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# %%
# create random inputs and outputs
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# %%
# randomly init weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

# %%
learning_rate = 1e-6
for t in range(500):
    # forward pass
    h = x.mm(w1)
    # torch.clamp(input, min, max, out=None) -> Tensor
    # 将input张量每个元素夹紧到[min,max]区间内
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    # print loss
    loss = (y_pred-y).pow(2).sum().item()
    print(t, loss)
    # backprop
    grad_y_pred = 2.*(y_pred-y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)
    # update weight
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2


## @@@@@@@@@@@@
## Autograd

#################################
## PyTorch: Tensors & autograd
# %%
dtype = torch.float
device = torch.device('cpu')
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# %%
# create random inputs and outputs
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# %%
# randomly init weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# %%
learning_rate = 1e-6
for t in range(500):
    # forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # print loss
    loss = (y_pred-y).pow(2).sum()
    if (t+1) % 100 == 0:
        # `.item()`获取单个元素Tensor的标量值
        print(t, loss.item())
    # backprop
    loss.backward()
    # update weight
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        # manually zero gradients
        w1.grad.zero_()
        w2.grad.zero_()


############################################
## PyTorch: Define new autograd functions
# %%
# 可以通过继承`torch.autograd.Function`来实现
# 自定义的autograd Functions
class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        ouputs = inputs.clamp(min=0)
        ctx.save_for_backward(inputs)
        return ouputs
    
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs[inputs<0] = 0
        return grad_inputs

# 可以通过继承`torch.autograd.Function`来实现
# 自定义的autograd Functions
class MyMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights):
        # N*I I*O  N*O
        ouputs = inputs.mm(weights)
        ctx.save_for_backward(inputs, weights)
        return ouputs
    
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, weights = ctx.saved_tensors
        grad_inputs = grad_outputs.mm(weights.t())
        grad_weights = inputs.t().mm(grad_outputs)
        return grad_inputs, grad_weights

# %%
dtype = torch.float
device = torch.device('cpu')
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# %%
# create random inputs and outputs
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# %%
# randomly init weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# %%
learning_rate = 1e-6
for t in range(500):
    # To apply custom Function, we use Function.apply method
    mm = MyMM.apply
    relu = MyReLU.apply
    # forward pass
    y_pred = mm(relu(mm(x,w1)),w2)
    # print loss
    loss = (y_pred-y).pow(2).sum()
    if (t+1) % 100 == 0:
        # `.item()`获取单个元素Tensor的标量值
        print(t, loss.item())
    # backprop
    loss.backward()
    # update weight
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        # manually zero gradients
        w1.grad.zero_()
        w2.grad.zero_()


## @@@@@@@@@@@@
## nn.Module

# nn.Module包定义了一组Modules，大致等效于神经网络的Layers，
# 对神经网络进行更高级别的抽象
# nn.Module包还定义了一组常用的Loss functions

#################
## PyTorch: nn
# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# %%
# create random inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# %%
# 将神经网络定义为Layers的序列（Sequence）
model = torch.nn.Sequential(OrderedDict([
    ('linear1', torch.nn.Linear(D_in, H)),
    ('relu', torch.nn.ReLU()),
    ('linear2', torch.nn.Linear(H, D_out)),
]))
print(model)

# %%
# 定义损失函数为均方差（Mean Squared Error, MSE）
# reduction = 'none'|'mean'|'sum'
loss_fn = torch.nn.MSELoss(reduction='sum')

# %%
learning_rate = 1e-4
for t in range(500):
    # forward pass
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    if (t+1) % 100 == 0:
        print(t, loss.item())
    # zero gradients
    model.zero_grad()
    # backprop
    loss.backward()
    # update weights
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad


####################
## PyTorch: optim
# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# create random inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 将神经网络定义为Layers的序列（Sequence）
model = torch.nn.Sequential(OrderedDict([
    ('linear1', torch.nn.Linear(D_in, H)),
    ('relu', torch.nn.ReLU()),
    ('linear2', torch.nn.Linear(H, D_out)),
]))
print(model)

# 定义损失函数为均方差（Mean Squared Error, MSE）
# reduction = 'none'|'mean'|'sum'
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # forward pass
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    if (t+1) % 100 == 0:
        print(t, loss.item())
    # zero gradients
    optimizer.zero_grad()
    # backprop
    loss.backward()
    # update weights
    optimizer.step()


################################
## PyTorch: Custom nn Modules
# %%
# 通过继承`torch.nn.Module`实现自定义Modules
# 定义一个简单双层网络模型 SimpleNet
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化模型，定义好Layers
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, inputs):
        # 重写`.forward()`方法，接收模型输入
        hidden = self.linear1(inputs)
        hidden_relu = hidden.clamp(min=0)
        y_pred = self.linear2(hidden_relu)
        return y_pred

# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# create random inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 网络模型
model = SimpleNet()
print(model)

# 定义损失函数为均方差（Mean Squared Error, MSE）
# reduction = 'none'|'mean'|'sum'
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # forward pass
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    if (t+1) % 100 == 0:
        print(t, loss.item())
    # zero gradients
    optimizer.zero_grad()
    # backprop
    loss.backward()
    # update weights
    optimizer.step()

############################################
## PyTorch: Control Flow + Weight Sharing
# %%
# 动态网络模型：由于PyTorch使用的是动态图，
# 因此可以轻易地实现动态网络模型，
# 即模型的网络结构在迭代里可以动态地变化
class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = torch.nn.Linear(D_in, H)
        self.mid_linear = torch.nn.Linear(H, H)
        self.out_linear = torch.nn.Linear(H, D_out)
    
    def forward(self, inputs):
        hidden_relu = self.in_linear(inputs).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            hidden_relu = self.mid_linear(hidden_relu).clamp(min=0)
        y_pred = self.out_linear(hidden_relu)
        return y_pred

# %%
# N: batch_size; D_in: dim of input
# H: dim of hidden; D_out: dim of output
N, D_in, H, D_out = 64, 1000, 100, 10

# create random inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 网络模型
model = DynamicNet()
print(model)

# 定义损失函数为均方差（Mean Squared Error, MSE）
# reduction = 'none'|'mean'|'sum'
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # forward pass
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    if (t+1) % 100 == 0:
        print(t, loss.item())
    # zero gradients
    optimizer.zero_grad()
    # backprop
    loss.backward()
    # update weights
    optimizer.step()

# %%
