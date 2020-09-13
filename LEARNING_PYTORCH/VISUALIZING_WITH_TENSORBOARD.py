## ================================== ##
##    VISUALIZING WITH TENSORBOARD    ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

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
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


## @@@@@@@@@@@@@@@@@@@@@@@@@@
## get data (Fashion-MNIST)

# %%
# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# datasets
trainset = torchvision.datasets.FashionMNIST(
                root='./data', 
                train=True, 
                transform=transform, 
                download=True)
testset = torchvision.datasets.FashionMNIST(
                root='./data', 
                train=False, 
                transform=transform, 
                download=True)

# dataloaders
trainloader = DataLoader(
                dataset=trainset, 
                batch_size=4, 
                shuffle=True, 
                num_workers=0)
testloader = DataLoader(
                dataset=testset, 
                batch_size=4, 
                shuffle=False, 
                num_workers=0)

# mapping of classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# %%
images, labels = iter(trainloader).next()
print(len(images), images[0].size(), labels[0])

# %%
# helper function
def matplot_imshow(hd, image, one_channel=False):
    if one_channel:
        image = image.squeeze(dim=0)
    image = image / 2 + 0.5
    if one_channel:
        hd.imshow(image.numpy(), cmap='gray')
    else:
        hd.imshow(np.transpose(image.numpy(), (1, 2, 0)))

print(classes[labels[0]])
matplot_imshow(plt, images[0], one_channel=True)


## @@@@@@@@@@@@@@@@@@@@@@@
## model + loss + optim

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

for param in net.named_parameters():
    print(param[0], param[1].size())

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


## @@@@@@@@@@@@@@
## TensorBoard

#######################
## TensorBoard setup

# %%
from torch.utils.tensorboard import SummaryWriter

# 默认的`log_dir`是`runs`，可自定义
writer = SummaryWriter(log_dir='runs/fashion_mnist_experiment_1')


############################
## Writing to TensorBoard

# %%
images, labels = iter(trainloader).next()
print(images.size())

# 为多图片显示添加图片网格 grid
img_grid = torchvision.utils.make_grid(images)
print(img_grid.size())

matplot_imshow(plt, img_grid)

# 使用`.add_image()`方法写入图片到TensorBoard
writer.add_image(tag='four_fashion-mnist_images', img_tensor=img_grid)

# %%
# Command: tensorboard --logdir=runs
# Visit: http://localhost:6006/


#########################################
## Inspect the model using TensorBoard

# %%
# 使用`.add_graph()`方法添加可视化模型结构
writer.add_graph(model=net, input_to_model=images)


#########################################
## Adding a "Projector" to TensorBoard

# %%
# helper function
def random_n_samples(data, labels, n=100):
    ''' sample n datapoints randomly
    '''
    assert len(data) == len(labels)
    assert n > 0
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# `.data` - 图片数据，`.targets` - 标签数据
images, labels = random_n_samples(trainset.data, trainset.targets)

# 标签映射
cls_labels = [classes[e] for e in labels]

# 获取高维特征
features = images.view(-1, 28*28)
# 使用`.add_embedding()`方法可视化高维数据的低维表示
writer.add_embedding(
                mat=features, 
                metadata=cls_labels, 
                label_img=images.unsqueeze(dim=1))


##############################################
## Tracking model training with TensorBoard

# %%
# helper function
def inputs_to_preds(net, inputs):
    outputs = net(inputs)
    _, indices = torch.max(outputs, dim=1)
    preds = indices.numpy()
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]
    return preds, probs

def plot_classes_preds(net, inputs, labels):
    preds, probs = inputs_to_preds(net, inputs)
    fig = plt.figure(figsize=(12,48))
    for i in np.arange(4):
        ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
        matplot_imshow(ax, inputs[i], one_channel=True)
        ax.set_title("{}, {:.2%}\n(label: {})".format(
            classes[preds[i]], 
            probs[i]*100.0, 
            classes[labels[i]]
        ), color=('green' if preds[i]==labels[i] else 'red'))
    return fig

# %%
# Training Loops
print("Start Train")

net.train()
running_loss = 0.
for epoch in range(1):
    for i, data in tqdm(enumerate(trainloader, start=1)):
        # get inputs/targets
        inputs, targets = data

        # zero gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 0:
            # 使用`add_scalar()`方法记录训练损失（或其他标量指标）
            writer.add_scalar(
                    tag='train_loss_1000_step', 
                    scalar_value=running_loss/1000, 
                    global_step=epoch*len(trainloader)+i)
            # 使用`.add_figure()`方法写入图片到TensorBoard
            # 与`.add_image()`的区别在于输入的形式，前者接收figure对象，后者接收tensor
            writer.add_figure(
                    tag='predictions vs. actuals', 
                    figure=plot_classes_preds(net, inputs, targets), 
                    global_step=epoch*len(trainloader)+i)
            running_loss = 0.

print("Finish Train")


###############################################
## Assessing trained models with TensorBoard

# %%
cls_preds = []
cls_probs = []
cls_targets = []
net.eval()
with torch.no_grad():
    for data in tqdm(testloader):
        inputs, targets = data
        outputs = net(inputs)
        batch_probs = F.softmax(outputs, dim=1)
        _, batch_preds = torch.max(batch_probs, dim=1)
        cls_probs.append(batch_probs)
        cls_preds.append(batch_preds)
        cls_targets.append(targets)

# 批次拼接
test_probs = torch.cat(cls_probs)
test_preds = torch.cat(cls_preds)
test_targets = torch.cat(cls_targets)

# %%
# sum(test_preds != test_targets)

# %%
# helper function
def add_pr_curve(cls_index, test_targets, test_probs, global_step=0):
    targets = test_targets == cls_index
    probs = test_probs[:, cls_index]
    # 使用`.add_pr_curve()`方法记录各类别的PR曲线
    writer.add_pr_curve(
                    tag='pr_curve: {}'.format(classes[cls_index]), 
                    labels=targets,     # 真实标签
                    predictions=probs,  # 分类概率
                    global_step=global_step)

# %%
# 为所有类别添加PR曲线
for i in range(len(classes)):
    add_pr_curve(i, test_targets, test_probs)

# %%
