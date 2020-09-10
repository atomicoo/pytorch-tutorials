## ================================== ##
## DL WITH PYTORCH: A 60 MINUTE BLITZ ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

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

torch.random.seed(0)

## @@@@@@@@@@
## 快速开始

##########
## 张量
# %%
# 利用函数方法创建
x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# %%
# 利用数据创建
x = torch.tensor([5.5, 3])
print(x)

# %%
# 利用已有Tensor创建
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

# %%
print(x.size()) # size
# or x.shape

##########
## 操作
# %%
y = torch.ones(5, 3)
print(x + y)
# or torch.add(x, y)
result = torch.empty(5, 3)
torch.add(x, y, out=result) # 指定输出的Tensor
print(result)

# %%
# 所有操作加上`_`后缀将成为改变原Tensor的操作
# 例如：x.opt_()会改变x本身，而x.opt()则不会
y.add(x)
print(y)
y.add_(x)
print(y)

# %%
# 可以使用类似NumPy的标准索引方式
print(x[:, 1])

# %%
# 使用`torch.view`改变Tensor形状
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # 与np.reshape()同，指定为-1的轴将自动计算其size
print(x.size(), y.size(), z.size())

# %%
# 对于单元素Tensor，使用`torch.item()`获取其数值
x = torch.randn(1)
print(x)
print(x.item())

# %%
# 关于Tensor的更多操作详见 https://pytorch.org/docs/stable/torch.html


## @@@@@@@@@@@@
## NumPy互转

# NumPy.Array与Torch.Tensor将共享底层存储，其一改变则一起改变
# 注意，前提是Torch.Tensor在CPU上

###################
## 转NumPy.Array
# %%
a = torch.ones(5)
print(a, type(a))
b = a.numpy()
print(b, type(b))

# %%
# 共享底层存储
b[::2] = 0
print(a)    # a的数据改变
a.add_(1)
print(b)    # b的数据改变

# %%
# 当a在GPU时不会共享底层存储，因为NumPy不支持GPU
# 或者说，a转移到GPU的操作是深拷贝
a = a.cuda()
b[::2] = 0
print(a)

####################
## 转Torch.Tensor
# %%
import numpy as np

# %%
a = np.ones(5)
print(a, type(a))
b = torch.from_numpy(a)
print(b, type(b))

# %%
# 共享底层存储
np.add(a, 1, out=a)
print(b)    # b的数据改变
b.add_(-1)
print(a)    # a的数据改变


## @@@@@@@@@@@@@@@
## CUDA Tensors

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# 可以在创建Tensor时直接指定设备为GPU，也可以
# 在创建Tensor之后使用`torch.to()`或者`torch.cuda()`进行转换
x = torch.randn(4, 4)
y = torch.ones_like(x, device=device)
x = x.to(device)
z = x + y
print(z)
z = z.to('cpu', torch.double)
print(z)

# %%
