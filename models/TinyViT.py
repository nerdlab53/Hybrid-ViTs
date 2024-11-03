import torch
import torch.nn as nn
from utils.patch_embedding import PatchEmbedding
from utils.initialization import init_vit_weights
from models.tiny_vit_blocks import TinyTransformerBlock

class TinyViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_channels=3,
        patch_size=32,
        embeddingdim=192,  # Reduced from original
        dropout=0.1,
        num_heads=3,  # Fixed at 3 as requested
        mlp_size=768,  # Reduced from original
        num_transformer_layer=4,  # Reduced from original
        num_classes=4
    ):
        super().__init__()
        
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        
        self.patch_embedding = PatchEmbedding(
            in_channels=num_channels,
            patch_size=patch_size,
            embeddingdim=embeddingdim
        )
        
        # Add batch norm after patch embedding
        self.embedding_bn = nn.BatchNorm1d(embeddingdim)
        
        num_patches = (img_size * img_size) // (patch_size * patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embeddingdim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embeddingdim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TinyTransformerBlock(
                embeddingdim,
                num_heads,
                mlp_size,
                dropout
            ) for _ in range(num_transformer_layer)
        ])
        
        self.norm = nn.LayerNorm(embeddingdim)
        
        # Two-layer MLP head with batch norm
        self.mlp_head = nn.Sequential(
            nn.Linear(embeddingdim, mlp_size),
            nn.BatchNorm1d(mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, num_classes)
        )
        
        # Initialize weights
        self.apply(init_vit_weights)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.pos_embedding)
        
        self.attention_weights = []

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Apply batch norm
        B = x.size(0)
        x = self.embedding_bn(x.transpose(1, 2)).transpose(1, 2)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding and dropout
        x = x + self.pos_embedding
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