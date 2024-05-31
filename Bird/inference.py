from dataset import get_dataloaders
from train import test
from torchvision import models
import torch
from model import create_resnet34

_, test_dataloader = get_dataloaders(root_dir='./CUB_200_2011/images/',batch_size=32)

model_pth = './logs/0530_lr_0.0003_decay_0.0003_pretrained_True_dropout_True_cutoutFalse_modelResNet34/best_model.pth'
model = create_resnet34(num_classes=200,pretrained=False)
checkpoint = torch.load(model_pth)
model.load_state_dict(checkpoint['parameters'])
model.to(device='cuda:0')
test(model,test_dataloader,device=0)

# Test: Acc:83.12128922815945