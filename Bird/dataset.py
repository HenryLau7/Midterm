import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from torchvision.datasets import ImageFolder



class CUB200Dataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.dataset)

def get_dataloaders(root_dir, batch_size=32):
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
    #     transforms.RandomRotation(20),            # 随机旋转±20度
    #     # Cutout(n_holes=5, length=16),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # test_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),        # 随机裁剪并调整到指定大小
    transforms.RandomHorizontalFlip(),        # 随机水平翻转，增加左右方向上的泛化
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
    transforms.RandomRotation(20),            # 随机旋转±20度
    transforms.ToTensor(),                    # 转换成Tensor
    Cutout(n_holes=5, length=16),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])
    test_transform = transforms.Compose([
        transforms.Resize(256),                   
        transforms.CenterCrop(224),               
        transforms.ToTensor(),                    
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])

    full_data = ImageFolder(root=root_dir, transform=None)

    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_dataset = CUB200Dataset(train_data, train_transform)
    test_dataset = CUB200Dataset(val_data, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


# dataset = CUB200Dataset(root_dir='./CUB_200_2011/',train=True,transform=None)
# print(dataset.__len__)


class Cutout(object):
    """ 随机遮挡图像中的一部分 """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 需要被处理的图像。
        Returns:
            PIL Image: 被遮挡后的图像。
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = np.clip(y - self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            y2 = np.clip(y + self.length // 2, 0, h)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
