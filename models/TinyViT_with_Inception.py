import torch
import torch.nn as nn
from models.inception_modules import TinyInceptionModule
from models.tiny_vit_blocks import TinyTransformerBlock
from utils.initialization import init_vit_weights

class TinyViT_with_Inception(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=32,
        in_channels=3,
        num_classes=4,
        dim=192,  # Reduced from original
        depth=4,   # Reduced from original
        num_heads=3,  # Fixed at 3
        mlp_dim=768,  # Reduced from original
        dropout=0.1
    ):
        super().__init__()
        
        # Tiny Inception module
        self.inception = TinyInceptionModule(in_channels=in_channels)
        
        # Calculate output size after inception module (128 channels)
        inception_out_channels = 128
        
        # Add a reduction layer after inception with batch norm
        self.reduction = nn.Sequential(
            nn.Conv2d(inception_out_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        # Position embeddings and class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TinyTransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        # MLP head with batch norm
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(init_vit_weights)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.pos_embed)
        
        self.attention_weights = []

    def forward(self, x):
        # Inception module
        x = self.inception(x)
        
        # Reduction layer
        x = self.reduction(x)
        
        # Reshape and transpose
        B = x.size(0)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding and dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Store attention weights
        self.attention_weights = []
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x, attn = block(x)
            self.attention_weights.append(attn)
        
        # Apply norm and get cls token
        x = self.norm(x)
        x = x[:, 0]
        
        # Classification head
        x = self.mlp_head(x)
        return x