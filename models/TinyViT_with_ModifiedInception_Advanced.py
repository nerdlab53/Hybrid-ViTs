import torch
import torch.nn as nn
from models.inception_modules import TinyModifiedInceptionModule
from .PretrainedTinyViTBase import PretrainedTinyViTBase
import torch._dynamo
from torch._dynamo import optimize

class TinyViT_with_ModifiedInception_Advanced(PretrainedTinyViTBase):
    def __init__(
        self,
        img_size=224,
        num_channels=3,
        patch_size=16,
        num_classes=4,
        dropout=0.1,
        freeze_backbone=True,
        gradient_checkpointing=True
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
        self.inception = TinyModifiedInceptionModule(in_channels=num_channels)
        
        # Add a reduction layer after inception
        self.reduction = nn.Sequential(
            nn.Conv2d(44, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU()
        )
        
        # Enable gradient checkpointing
        self.gradient_checkpointing = gradient_checkpointing
        
    def forward(self, x):
        # Apply inception module
        x = self.inception(x)
        
        # Reduce channels back to original input size
        x = self.reduction(x)
        
        # Get the pretrained backbone
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(super().forward, x)
        return super().forward(x)
        
    def get_layer_groups(self):
        return [
            {'params': self.inception.parameters(), 'lr_mult': 5.0},  # Highest LR
            {'params': self.reduction.parameters(), 'lr_mult': 3.0},  # High LR
            {'params': self.backbone.parameters(), 'lr_mult': 1.0},  # Base LR
            {'params': self.head.parameters(), 'lr_mult': 2.0}  # Medium LR
        ]