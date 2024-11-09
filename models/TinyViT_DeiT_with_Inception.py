import torch
import torch.nn as nn
from models.inception_modules import TinyInceptionModuleLite
from .PretrainedTinyViTBase import PretrainedTinyViTBase

class TinyViT_DeiT_with_Inception(PretrainedTinyViTBase):
    def __init__(
        self,
        img_size=224,
        num_channels=3,
        patch_size=16,
        num_classes=4,
        dropout=0.1,
        freeze_backbone=True
    ):
        super().__init__(
            pretrained_model_name='deit_tiny_patch16_224',
            img_size=img_size,
            num_channels=num_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    
        self.inception = TinyInceptionModuleLite(in_channels=num_channels)
        
        self.reduction = nn.Sequential(
            nn.Conv2d(32, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.inception(x)
        
        x = self.reduction(x)
        
        return super().forward(x) 