import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Branch 1: 1x1 conv
        self.branch1x1_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch3x3_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 conv -> 5x5 conv
        self.branch5x5_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(64, 96, kernel_size=5, padding=2)
        
        # Branch 4: max pool -> 1x1 conv
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(self, x):
        # Branch 1
        branch1x1 = F.relu(self.bn1(self.branch1x1_1(x)))
        
        # Branch 2
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.bn2(self.branch3x3_2(branch3x3)))
        
        # Branch 3
        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.bn3(self.branch5x5_2(branch5x5)))
        
        # Branch 4
        branch_pool = self.branch_pool(x)
        branch_pool = F.relu(self.bn4(self.branch_pool_1(branch_pool)))
        
        # Concatenate along channel dimension
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)