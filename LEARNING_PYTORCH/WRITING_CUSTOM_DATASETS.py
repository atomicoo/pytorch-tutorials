## ================================== ##
##       WRITING CUSTOM DATASETS      ##
## ================================== ##

## 参考资料
#  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#  http://pytorch123.com/ThirdSection/DataLoding/

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
from torch.utils.data import Dataset, DataLoader

import os
from pathlib import Path
import requests
import zipfile
import numpy as np
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets


######################################################
## 下载解压数据文件
###
# %%
DATA_PATH = Path("data")
PATH = DATA_PATH / "faces"

URL = "https://download.pytorch.org/tutorial/"
FILENAME = "faces.zip"

if not (DATA_PATH / FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (DATA_PATH / FILENAME).open('wb').write(content)
else:
    print("## [INFO] data zip file already exists.".format(FILENAME))

if not PATH.exists():
    with zipfile.ZipFile((DATA_PATH / FILENAME), 'r') as zf:
        zf.extractall(DATA_PATH)
else:
    print("## [INFO] data directory already exists.")


######################################################
## 探索数据集
###
# 数据集打标说明：
# 数据集打标结果按照如下格式组织为.csv文件
# image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
# 0805personali01.jpg,27,83,27,98, ... 84,134
# 1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
# %%
dframe = pd.read_csv((PATH / "face_landmarks.csv"))

n = 11
img_name = dframe.iloc[n, 0]
# 数据格式转换，numpy.ndarray
landmarks = dframe.iloc[n, 1:].values
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: \n{}'.format(landmarks[:4]))

# %%
# helper function，显示打标图片
def show_landmarks(hd, image, landmarks):
    hd.imshow(image)
    hd.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='r', marker='.')

plt.figure()
show_landmarks(plt, io.imread((PATH / img_name)), landmarks)
plt.show()

######################################################
## 数据集抽象类：Dataset
###
# `torch.utils.data.Dataset`为PyTorch的数据集抽象类，
# 子类化时需要重写：
#   - `__len__`方法实现`len(dataset)`时返回数据集尺寸
#   - `__getitem__`方法实现`dataset[i]`时返回索引数据
# %%
class FacesDataset(Dataset):
    """
    Faces Landmarks Dataset.
    """
    def __init__(self, csv_path, root_dir, transform=None):
        super().__init__()
        self.dframe = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dframe)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        img_name = self.dframe.iloc[index, 0]
        # numpy.ndarray: H x W x C
        image = io.imread(os.path.join(self.root_dir, img_name))
        landmarks = self.dframe.iloc[n, 1:].values
        # numpy.ndarray: N x 2
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        # transform
        if self.transform:
            sample = self.transform(sample)
        return sample

# %%
# 实例化自定义数据集对象FacesDataset
faces_ds = FacesDataset((PATH / "face_landmarks.csv"), PATH)
print("Dataset length: {}".format(len(faces_ds)))
sample = faces_ds[11]
print("Show landmarks: \n")
show_landmarks(plt, **sample)


######################################################
## 数据转换：Transform
###
# 大多数神经网络默认输入尺寸相同，因此这里我们需要对图片进行
# 缩放、随机裁剪等操作来确保尺寸一致，这同时也是一种数据增强的方式
# 将其写成可调用类（而非普通的函数）是非常明智的，只需要实现`__call__`方法，
# 并在必要时实现`__init__`方法即可
# 
# tsfm = Transform(params)
# transformed_sample = tsfm(sample)
# %%
# 图片缩放操作可调用类
class Rescale(object):
    """Rescale the image to a given size.

    Args:
        output_size (tuple or int): Desired output size.
            If tuple, output is matched to output_size.
            If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        super().__init__()
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        # 获取图片及打标点
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        # 获取输出尺寸
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size/w*h, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size/h*w
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # 图片缩放
        image = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [ new_h/h, new_w/w ]    # 利用广播机制
        return { 'image': image, 'landmarks': landmarks }

# %%
# 图片随机裁剪操作可调用类
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """
    def __init__(self, output_size):
        super().__init__()
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        image = image[top:top+new_h, left:left+new_w]
        landmarks = landmarks - [ left, top ]
        return { 'image': image, 'landmarks': landmarks }

# %%
# 图片数据类型转换可调用类
# 从numpy.ndarray转为torch.tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap axis order
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image, landmarks = torch.from_numpy(image), torch.from_numpy(landmarks)
        return { 'image': image, 'landmarks': landmarks }

# %%
# 
sample = faces_ds[11]
print(sample['image'].shape)
S = Rescale(256)
sample = S(sample)
print(sample['image'].shape)
C = RandomCrop(128)
sample = C(sample)
print(sample['image'].shape)

# show transformed image
show_landmarks(plt, **sample)

T = ToTensor()
sample = T(sample)
print(sample['image'].shape)

# %%
# Compose transforms
composed = transforms.Compose([
                Rescale(256), RandomCrop(224)])

# %%
# Transformed Dataset
faces_ds = FacesDataset(
                csv_path=(PATH / "face_landmarks.csv"), 
                root_dir=PATH, 
                transform=transforms.Compose([
                    Rescale(256), RandomCrop(224), ToTensor()
                ]))

print("Dataset length: {}".format(len(faces_ds)))
sample = faces_ds[11]
print("Sample info: ")
print("\tImage size: {}".format(sample['image'].size()))
print("\tLandmarks size: {}".format(sample['landmarks'].size()))

image = sample['image'].numpy().transpose((1, 2, 0))
landmarks = sample['landmarks'].numpy()
show_landmarks(plt, image, landmarks)


######################################################
## 遍历数据集：DataLoader
###
# `torch.utils.data.DataLoader`联合了Dataset与Sampler，
# Dataset为数据集，Sampler为数据获取策略
# %%
# 循环遍历
# 显然很麻烦，而且放弃了批处理、多线程等操作的优势
for i in range(len(faces_ds)):
    sample = faces_ds[i]
    print(i, sample['image'].size(), sample['landmarks'].size())
    break

# %%
# DataLoader
faces_dl = DataLoader(dataset=faces_ds, batch_size=4,
                      shuffle=True, num_workers=0)

# %%
# helper function to show batch images
def show_landmarks_batch(hd, batched_samples):
    batched_images, batched_landmarks = \
        batched_samples['image'], batched_samples['landmarks']
    batch_size = len(batched_images)
    image_size = batched_images.size(3)
    print(image_size)
    grid_border_size = 2
    grided_images = utils.make_grid(batched_images)
    # show images
    hd.imshow(grided_images.numpy().transpose((1, 2, 0)))
    # show landmarks
    for i in range(batch_size):
        hd.scatter(
            x=batched_landmarks[i, :, 0].numpy()+i*image_size+(i+1)*grid_border_size,
            y=batched_landmarks[i, :, 1].numpy()+grid_border_size,
            s=10, c='r', marker='.'
        )
    hd.title('Batch from dataloader')

# %%
batched_samples = iter(faces_dl).next()
show_landmarks_batch(plt, batched_samples)


######################################################
## Afterword: torchvision
###
# `torchvision`包中封装了很多常用的图像数据集和转换操作，
# `torchvision`包甚至封装了一个更为通用的数据集类`ImageFolder`，
# 其假定图像数据集组织方式如下：
# 
# root/ants/xxx.png
# root/ants/xxy.jpeg
# ... ...
# root/bees/123.jpg
# root/bees/nsdf3.png
# ... ...
# %%
# 使用torchvision包方便地调用各种封装好的类来操作图像数据
data_transform = transforms.Compose([
        # transforms.Scale()
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(dataset=hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

# %%
