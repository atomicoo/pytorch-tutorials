## ================================== ##
## DL WITH PYTORCH: A 60 MINUTE BLITZ ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

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


## @@@@@@@@@@@
## 神经网络

#############
## 定义网络
# %%
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义全连接层
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 1x32x32 -> 6x30x30 -> 6x15x15
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # 6x15x15 -> 16x13x13 -> 16x6x6
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 16x6x6 -> 576
        x = x.view(-1, self.num_flat_feats(x))
        # 576 -> 120
        x = F.relu(self.fc1(x))
        # 120 -> 84
        x = F.relu(self.fc2(x))
        # 84 -> 10
        x = self.fc3(x)
        return x
    
    def num_flat_feats(self, x):
        sz = x.size()[1:]
        num = 1
        for s in sz: num *= s
        return num

# %%
net = Net()
print(net)

# %%
# 可学习参数
params = list(net.parameters())
print(len(params))
for param in net.named_parameters():
    print(param[0], param[1].size())

# %%
# 随机输入
inp = torch.randn(4, 1, 32, 32)
outp = net(inp)
print(outp)

# %%
# 反向传播之前需要先将梯度缓冲区清零
net.zero_grad()
outp.backward(torch.randn(1, 10))

# %%
# torch.nn仅支持批次输入，因此nn.Conv2D()接收4-D张量(nSamples x nChannels x Height x Width)
# 其中第一个维度是批次尺寸。当仅有一条数据时可以调用`.unsqueeze(0)`生成“伪轴”

# %%
# torch.Tensor: 多维数组，支持autograd操作，并保存梯度
# nn.Module: 神经网络模块。封装参数及移动到设备、导出、加载等辅助方法
# nn.Parameter: 一种Tensor，将其作为属性分配给Module时会自动注册为参数
# autograd.Function: 实现autograd操作的前/后向定义。每个Tensor操作都至少创建一个Function节点，该节点连接到创建Tensor的函数并编码其历史


#############
## 损失函数
# %%
outp = net(inp)
tgt = torch.randn(10).view(1, -1)
criterion = nn.MSELoss()

# %%
loss = criterion(outp, tgt)
print(loss)

# %%
# 反向传播过程：
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


#############
## 反向传播
# %%
# 反向传播之前需要先将梯度缓冲区清零
net.zero_grad()
print('conv1.bias.grad before backward')
# 根据名称获取参数：`.conv1.weight`、`.conv1.bias`
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
# 根据名称获取参数：`.conv1.weight`、`.conv1.bias`
print(net.conv1.bias.grad)


#############
## 权重更新
# %%
# 手动更新
learning_rate = 0.01
for param in net.parameters():
    # 注意！！！在`param.data`上操作而不是`param`
    param.data.sub_(param.grad.data * learning_rate)

# %%
param = next(net.parameters())
# `param`与`param.data`
print(param.requires_grad)
print(param.data.requires_grad)
# or param.detach().requires_grad

# %%
param = net.conv1.bias
print('conv1.bias before update')
print(param)
param.data.sub_(param.grad.data * 0.1)
print('conv1.bias after update')
print(param)

# %%
# torch.optim包 - 优化器
# 创建优化器，需要传入网络参数
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 训练Loops中使用优化器
optimizer.zero_grad()   # 所有梯度清零（包括网络中的参数）
outp = net(inp)
loss = criterion(outp, tgt)
loss.backward()     # 反向传播
optimizer.step()    # 更新权重

# %%
