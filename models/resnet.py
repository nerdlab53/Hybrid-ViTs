import torch
import torch.nn as nn
from torchvision import models
from dataset_utils.config import Alzheimer_CFG

config = Alzheimer_CFG()

class ResNet50_for_Alzheimer(nn.Module):

    def __init__(self, num_classes=config.num_classes):
        super(ResNet50_for_Alzheimer, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        prev_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(prev_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
