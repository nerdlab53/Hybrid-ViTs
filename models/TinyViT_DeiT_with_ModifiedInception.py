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
        # First initialize the base class to load pretrained weights
        super().__init__(
            pretrained_model_name='deit_tiny_patch16_224',
            img_size=img_size,
            num_channels=num_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        # Then add modified inception module
        self.inception = TinyModifiedInceptionModuleLite(in_channels=self.embed_dim)
        
        # Modified reduction layer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(44),  # Match inception output channels
            nn.Linear(44, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        # Get features from backbone
        x = self.backbone.forward_features(x)
        
        # Handle different output formats
        if len(x.shape) == 4:  # CNN-like output [B, C, H, W]
            x = self.global_pool(x)
        elif len(x.shape) == 3:  # Transformer-like output [B, N, C]
            x = x.mean(dim=1)  # Global average pooling over patches
        
        # Reshape for inception
        B = x.shape[0]
        x = x.view(B, self.embed_dim, 1, 1)
        
        # Apply inception
        x = self.inception(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Apply classification head
        x = self.mlp_head(x)
        return x