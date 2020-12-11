## ================================== ##
## DL WITH PYTORCH: A 60 MINUTE BLITZ ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

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

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


## @@@@@@@@@@@
## 关于数据
# CIFAR-10 Dataset
# Class: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’
# Sizes: 3 * 32 * 32

# `torchvision.datasets`中集成常用视觉数据集
# `torch.utils.data.DataLoader`
# `torch.utils.data.Dataset`


## @@@@@@@@@@@@@@@@
## 训练图片分类器

#############
## 导入数据
# %%
# `torchvision.datasets`输出为PILImage格式数据，需要进行数据归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

# %%
# CIFAR-10训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

# %%
# CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4,
                                         shuffle=False, num_workers=0)

# %%
# CIFAR-10类别映射
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck')


#############
## 探索数据
# %%
images, labels = iter(trainloader).next()
print(len(images), images[0].size(), labels[0])

# %%
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # plt.imshow() W x H x C
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %%
images, labels = iter(trainloader).next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%12s' % classes[labels[j]] for j in range(4)))


#############
## 神经网络
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


####################
## 损失函数与优化器
# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#############
## 训练模型
# %%
print('Start Training')
for epoch in range(2):
    running_loss = 0.
    for i, data in enumerate(trainloader, start=1):
        # 获取一个Batch的数据，data = (images, labels)
        inputs, labels = data
        # 所有梯度缓冲区清零
        optimizer.zero_grad()
        # forward
        outputs = net(inputs)
        # loss
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # update
        optimizer.step()
        # 打印各种指标
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[{:d}, {:5d}] loss: {:.3f}'.format
                  (epoch+1, i, running_loss/2000))
            running_loss = 0.0
print('Finish Training')

# %%
# 法一：仅保存模型参数
torch.save(net.state_dict(), './cifar_net.pth')
# 法二：保存完整模型
# torch.save(net, './cifar_net.pth')

# %%
# 法一：仅加载模型参数
net = Net()
net.load_state_dict(torch.load('./cifar_net.pth'))
# 法二：加载完整模型
# net = torch.load('./cifar_net.pth')


#############
## 测试模型
# %%
images, labels = iter(testloader).next()

# %%
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ',
      ' '.join('{:5s}'.format(classes[labels[j]] for j in range(4))))

# %%
outputs = net(images)
_, predicted = torch.max(outputs, axis=1)
print('Predicted: ',
      ' '.join('{:5s}'.format(classes[predicted[j]] for j in range(4))))

# %%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the {:d} test images: {:d}%'
      .format(total, 100*correct/total))


## @@@@@@@@@@@@
## 单GPU训练

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# 注意，使用GPU训练时，模型与数据都需要转到GPU中
net.to(device)
inputs, labels = data[0].to(device), data[1].to(device)

# %%
