import torch
import torch.nn as nn
from torchvision import models


def create_resnet34(num_classes=200,pretrained=True,dropout=True):
    model = models.resnet34(weights = pretrained)

    num_features = model.fc.in_features  
    
    if dropout:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 添加 Dropout
            nn.Linear(num_features, num_classes)
        )
    else:
        model.fc = nn.Linear(num_features, num_classes) 

    return model


def create_resnet18(num_classes=200,pretrained=True,dropout=True):
    model = models.resnet18(weights = pretrained)

    num_features = model.fc.in_features  
    
    if dropout:
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Dropout(p=0.5),  # 添加 Dropout
        )
    else:
        model.fc = nn.Linear(num_features, num_classes) 

    return model