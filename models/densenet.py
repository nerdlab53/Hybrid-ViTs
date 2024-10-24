import torch
import torch.nn as nn
from torchvision import models
from dataset_utils.config import Alzheimer_CFG

config = Alzheimer_CFG()

class DenseNet_for_Alzheimer(nn.Module):
    def __init__(self, num_classes=config.num_classes):
        super(DenseNet_for_Alzheimer, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)
