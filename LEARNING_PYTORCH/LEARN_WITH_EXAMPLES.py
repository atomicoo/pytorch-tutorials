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
for t in range(500, start=1):
    # forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # print loss
    loss = (y_pred-y).pow(2).sum().item()
    if t % 100 == 0:
        print(t, loss)
    # backprop
    loss.backward()
    # update weight
    with torch.no_grad():
        w1 -= learning_rate*grad_w1
        w2 -= learning_rate*grad_w2
        # manually zero the grad
        w1.grad.zero_()
        w2.grad.zero_()

# %%
