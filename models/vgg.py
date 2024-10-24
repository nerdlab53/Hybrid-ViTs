import torch
import torch.nn as nn
from torchvision import models
from dataset_utils.config import Alzheimer_CFG

config = Alzheimer_CFG()

class VGG_for_Alzheimer(nn.Module):
    def __init__(self, num_classes=config.num_classes):
        super(VGG_for_Alzheimer, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True)
        num_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.vgg(x)
