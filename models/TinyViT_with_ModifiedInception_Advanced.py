import torch
import torch.nn as nn
from models.inception_modules import TinyModifiedInceptionModule
from .TinyViT_with_ModifiedInception import TinyViT_with_ModifiedInception

class TinyViT_with_ModifiedInception_Advanced(TinyViT_with_ModifiedInception):
    def __init__(
        self,
        img_size=224,
        patch_size=32,
        in_channels=3,
        num_classes=4,
        dim=192,
        depth=4,
        num_heads=3,
        mlp_dim=768,
        dropout=0.1,
        gradient_checkpointing=True
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # Enable gradient checkpointing
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.transformer_blocks.gradient_checkpointing_enable()
            
    def get_layer_groups(self):
        return [
            {'params': self.inception.parameters(), 'lr_mult': 5.0},
            {'params': self.linear_proj.parameters(), 'lr_mult': 3.0},
            {'params': self.transformer_blocks.parameters(), 'lr_mult': 1.0},
            {'params': self.mlp_head.parameters(), 'lr_mult': 2.0}
        ] 