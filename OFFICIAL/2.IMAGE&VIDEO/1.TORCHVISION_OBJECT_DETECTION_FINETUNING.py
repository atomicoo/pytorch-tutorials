# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
# 
# ## 参考资料
# 
# > https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html  
# > Dataset: https://www.cis.upenn.edu/~jshi/ped_html/  
# > Model (Mask R-CNN): https://arxiv.org/abs/1703.06870  
# 

# %%
import os

import torch
print(torch.__version__)

# %% [markdown]
# ## 定义数据集
# 
# 自定义数据集应继承自标准 `torch.utils.data.Dataset` 类，并实现 `__len__` 和 `__getitem__` 方法。
# 
# 针对特定任务数据集只需注意相应的 `__getitem__` 方法，譬如本例中的对象检测数据集：
# 
# - 图像：尺寸为 `(H x W)` 的PIL图像
# - 目标：包含以下字段
#   - `boxes (FloatTensor[N, 4])`：`N` 个 `[x0, y0, x1, y1]`
#   - `labels (Int64Tensor[N])`：`0` 代表背景类
#   - `image_id (Int64Tensor[1])`：图像ID
#   - `area (Tensor[N])`：边界框面积，用于区分不同大小边界框的得分
#   - `iscrowd (UInt8Tensor[N])`：`iscrowd=True` 的对象在评估时将被忽略
#   - （可选）`masks (UInt8Tensor[N, H, W])`：每个对象的分割蒙版
#   - （可选）`keypoints (FloatTensor[N, K, 3])`：每个对象包含的K个`[x, y, visibility]`形式的关键点
# 
# 注意：Win10 下安装 `pycocotools` 请参考 https://blog.csdn.net/qq_28400629/article/details/85247087
# 

# %%
import numpy as np
from PIL import Image

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 省略代码：加载所有图像并排序，检查图像一致性
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 加载图像和蒙版
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 蒙版不转为RGB，因为之后每个对象需要对应不同的蒙版图像
        mask = Image.open(mask_path)
        # 将PIL图像转为Numpy.Array
        mask = np.array(mask)
        # 不同对象对应不同颜色
        obj_ids = np.unique(mask)
        # 第一个ID对应背景，此处不需要
        obj_ids = obj_ids[1:]

        # 因为一张图像会有多个对象，每个对象需要对应一个蒙版
        masks = (mask == obj_ids[:, None, None])

        # 获取每个对象蒙版的边界
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 转为`torch.Tensor`
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 本例中只有一个类别
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假定所有实例满足`iscrowd=False`
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # 构造`target`字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# %% [markdown]
# ## 定义模型
# 
# 本例使用基于 [Faster R-CNN](https://arxiv.org/abs/1506.01497) 的 [Mask R-CNN](https://arxiv.org/abs/1703.06870)。
# 
# ![intermediate/../../_static/img/tv_tutorial/tv_image03.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image03.png)
# 
# [Mask R-CNN](https://arxiv.org/abs/1703.06870) 额外增加了一个分支用于预测蒙版（Mask）：
# 
# ![intermediate/../../_static/img/tv_tutorial/tv_image04.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image04.png)
# 
# 下述两种常见情况：
# 
# %% [markdown]
# ### 1-预训练模型
# 

# %%
# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # 加载在COCO数据集上预训练过的模型
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# # 替换模型的头（分类器部分）
# # 根据需求设置类别数 `num_classes`
# num_classes = 2  # 1-person, 0-background
# # 分类器的输入特征 `in_features`
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# # 替换预训练模型的头（head）
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# %% [markdown]
# ### 2-添加其他分支
# 

# %%
# import torchvision
# from torchvision.models.detection.faster_rcnn import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator

# # 加载一个只会返回抽取特征的预训练分类模型
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# # FasterRCNN 需要知道输出通道数 `output channels`
# # 对于 mobilenet_v2 模型，`output channels=1280`
# backbone.out_channels = 1280

# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))

# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                 output_size=7,
#                                                 sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes=2,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)

# %% [markdown]
# ### 实例分割模型
# 

# %%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# %% [markdown]
# ## 辅助脚本
# 
# 一些帮助简化数据处理与训练/测试过程的辅助脚本：
# 
# - [coco_eval.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/coco_eval.py)
# - [coco_utils.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/coco_utils.py)
# - [engine.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/engine.py)
# - [group_by_aspect_ratio.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/group_by_aspect_ratio.py)
# - [train.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/train.py)
# - [transforms.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/transforms.py)
# - [utils.py](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/utils.py)

# %%
from references.detection import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    # 注意：这里将`.ToTensor()`放到`.RandomHorizontalFlip()`前
    # 会导致报错 `TypeError: img should be PIL Image. Got <class `torch.Tensor`>`
    return T.Compose(transforms)

# %% [markdown]
# ## 测试 `forward()`
# 
# 

# %%
# from references.detection import utils

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
# data_loader = torch.utils.data.DataLoader(dataset, 
#                                           batch_size=2, shuffle=True, 
#                                           num_workers=0, collate_fn=utils.collate_fn)


# %%
# # 训练 Training
# images,targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images, targets)   # 返回损失

# # 推断 inference
# model.eval()   # 评估模式
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)   # 返回预测值


# %%
from references.detection.engine import train_one_epoch, evaluate
from references.detection import utils

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


# %%
main()


# %%



