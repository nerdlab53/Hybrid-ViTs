import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedInceptionModule(nn.Module):
    '''Modified Inception Module implementation | Used as feature extractor for the ViTs
    '''
    def __init__(self, in_channels: int):
        super(ModifiedInceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_bn = nn.BatchNorm2d(64)
        self.branch5x5_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.branch5x5_bn = nn.BatchNorm2d(64)
        self.branch3x3_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.branch3x3_bn = nn.BatchNorm2d(96)
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch_pool1_bn = nn.BatchNorm2d(32)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.branch1x1_bn(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_bn(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch3x3 = self.branch3x3_bn(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch_pool = self.branch_pool1_bn(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)
    

class InceptionModule(nn.Module):
    '''Inception Module taken from the `Going deeper with convolutions` paper
    '''
    def __init__(self, in_channels: int):
        self.branch1x1_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_2_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_2_2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.branch1x1_3_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_3_2 = nn.Conv2d(96, 64, kernel_size=5, padding=2)
        self.branch_pool1 = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        branch1x1_1 = self.branch1x1_1(x)

        branch1x1_2_1 = self.branch1x1_2_1(x)
        branch1x1_2_2 = self.branch1x1_2_2(branch1x1_2_1)

        branch1x1_3_1 = self.branch1x1_3_1(x)
        branch1x1_3_2 = self.branch1x1_3_2(branch1x1_3_1)

        branch_pool = F.avg_pool(x, kernel_size=3, stride=1, padding=1)
        
        outputs = [branch1x1_1, branch1x1_2_2, branch1x1_3_2, branch_pool]
        return torch.cat(outputs, 1)