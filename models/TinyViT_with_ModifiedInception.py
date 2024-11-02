import torch
import torch.nn as nn
from models.inception_modules import TinyModifiedInceptionModule
from models.tiny_vit_blocks import TinyTransformerBlock
from utils.initialization import init_vit_weights

class TinyViT_with_ModifiedInception(nn.Module):
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
        
        # Modified Tiny Inception module
        self.inception = TinyModifiedInceptionModule(in_channels=in_channels)
        
        # Flatten and project with batch norm
        self.flatten = nn.Flatten(start_dim=1)
        
        # Calculate input dimension for linear projection (176 channels from TinyModifiedInception)
        input_dim = 176 * (img_size // patch_size) * (img_size // patch_size)
        
        self.linear_proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Position embeddings and class token
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TinyTransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # MLP head with batch norm
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
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
        B = x.shape[0]
        
        # Inception feature extraction
        x = self.inception(x)
        
        # Flatten and project
        x = self.flatten(x)
        x = self.linear_proj(x)
        x = x.view(B, -1, x.size(-1))
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Store attention weights
        self.attention_weights = []
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x, attn = block(x)
            self.attention_weights.append(attn)
        
        # Classification head
        x = x[:, 0]
        x = self.mlp_head(x)
        
        return x 