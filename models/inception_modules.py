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

class ModifiedInceptionModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Modified Branch 1: 1x1 conv with more filters
        self.branch1x1_1 = nn.Conv2d(in_channels, 96, kernel_size=1)
        
        # Modified Branch 2: 1x1 conv -> 3x3 conv with increased channels
        self.branch3x3_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        
        # Modified Branch 3: 1x1 conv -> two 3x3 convs
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        # Modified Branch 4: avg pool -> 1x1 conv
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3a = nn.BatchNorm2d(96)
        self.bn3b = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Branch 1
        branch1x1 = F.relu(self.bn1(self.branch1x1_1(x)))
        branch1x1 = self.dropout(branch1x1)
        
        # Branch 2
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.bn2(self.branch3x3_2(branch3x3)))
        branch3x3 = self.dropout(branch3x3)
        
        # Branch 3
        branch3x3dbl = F.relu(self.branch3x3dbl_1(x))
        branch3x3dbl = F.relu(self.bn3a(self.branch3x3dbl_2(branch3x3dbl)))
        branch3x3dbl = F.relu(self.bn3b(self.branch3x3dbl_3(branch3x3dbl)))
        branch3x3dbl = self.dropout(branch3x3dbl)
        
        # Branch 4
        branch_pool = self.branch_pool(x)
        branch_pool = F.relu(self.bn4(self.branch_pool_1(branch_pool)))
        branch_pool = self.dropout(branch_pool)
        
        # Concatenate along channel dimension
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
    
class TinyInceptionModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Branch 1: 1x1 conv with fewer filters
        self.branch1x1_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # Branch 2: 1x1 conv -> 3x3 conv with fewer channels
        self.branch3x3_1 = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 conv -> 5x5 conv with fewer channels
        self.branch5x5_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(32, 48, kernel_size=5, padding=2)
        
        # Branch 4: max pool -> 1x1 conv with fewer channels
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(16)

    def forward(self, x):
        branch1x1 = F.relu(self.bn1(self.branch1x1_1(x)))
        
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.bn2(self.branch3x3_2(branch3x3)))
        
        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.bn3(self.branch5x5_2(branch5x5)))
        
        branch_pool = self.branch_pool(x)
        branch_pool = F.relu(self.bn4(self.branch_pool_1(branch_pool)))
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)  # Total channels: 128 (32+32+48+16)

class TinyModifiedInceptionModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Branch 1: 1x1 conv
        self.branch1x1_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        
        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch3x3_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 conv -> two 3x3 convs
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        
        # Branch 4: avg pool -> 1x1 conv
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3a = nn.BatchNorm2d(48)
        self.bn3b = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.dropout = nn.Dropout(0.1)  # Reduced dropout rate

    def forward(self, x):
        branch1x1 = self.dropout(F.relu(self.bn1(self.branch1x1_1(x))))
        
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = self.dropout(F.relu(self.bn2(self.branch3x3_2(branch3x3))))
        
        branch3x3dbl = F.relu(self.branch3x3dbl_1(x))
        branch3x3dbl = F.relu(self.bn3a(self.branch3x3dbl_2(branch3x3dbl)))
        branch3x3dbl = self.dropout(F.relu(self.bn3b(self.branch3x3dbl_3(branch3x3dbl))))
        
        branch_pool = self.branch_pool(x)
        branch_pool = self.dropout(F.relu(self.bn4(self.branch_pool_1(branch_pool))))
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)  # Total channels: 176 (48+48+48+32)

class TinyInceptionModuleLite(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Branch 1: 1x1 conv with fewer filters
        self.branch1x1_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        
        # Branch 2: 1x1 conv -> 3x3 conv with fewer channels
        self.branch3x3_1 = nn.Conv2d(in_channels, 6, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 conv -> 5x5 conv with fewer channels
        self.branch5x5_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(8, 12, kernel_size=5, padding=2)
        
        # Branch 4: max pool -> 1x1 conv with fewer channels
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(12)
        self.bn4 = nn.BatchNorm2d(4)

    def forward(self, x):
        branch1x1 = F.relu(self.bn1(self.branch1x1_1(x)))
        
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.bn2(self.branch3x3_2(branch3x3)))
        
        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.bn3(self.branch5x5_2(branch5x5)))
        
        branch_pool = self.branch_pool(x)
        branch_pool = F.relu(self.bn4(self.branch_pool_1(branch_pool)))
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)  # Total channels: 32 (8+8+12+4)

class TinyModifiedInceptionModuleLite(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Branch 1: 1x1 conv
        self.branch1x1_1 = nn.Conv2d(in_channels, 12, kernel_size=1)
        
        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch3x3_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 conv -> two 3x3 convs
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        
        # Branch 4: avg pool -> 1x1 conv
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3a = nn.BatchNorm2d(12)
        self.bn3b = nn.BatchNorm2d(12)
        self.bn4 = nn.BatchNorm2d(8)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        branch1x1 = self.dropout(F.relu(self.bn1(self.branch1x1_1(x))))
        
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = self.dropout(F.relu(self.bn2(self.branch3x3_2(branch3x3))))
        
        branch3x3dbl = F.relu(self.branch3x3dbl_1(x))
        branch3x3dbl = F.relu(self.bn3a(self.branch3x3dbl_2(branch3x3dbl)))
        branch3x3dbl = self.dropout(F.relu(self.bn3b(self.branch3x3dbl_3(branch3x3dbl))))
        
        branch_pool = self.branch_pool(x)
        branch_pool = self.dropout(F.relu(self.bn4(self.branch_pool_1(branch_pool))))
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)  # Total channels: 44 (12+12+12+8)