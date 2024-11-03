import torch
import torch.nn as nn
from models.inception_modules import TinyModifiedInceptionModuleLite
from .PretrainedTinyViTBase import PretrainedTinyViTBase

class TinyViT_DeiT_with_ModifiedInception(PretrainedTinyViTBase):
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
        
        # Add Modified Inception module before the backbone
        self.inception = TinyModifiedInceptionModuleLite(in_channels=num_channels)
        
        # Add a reduction layer after inception
        self.reduction = nn.Sequential(
            nn.Conv2d(44, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU()
        )

    def forward(self, x):
        # Apply inception module
        x = self.inception(x)
        
        # Reduce channels back to original input size
        x = self.reduction(x)
        
        # Continue with normal forward pass
        return super().forward(x) 