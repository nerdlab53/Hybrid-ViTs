import torch 
import torch.nn as nn
import torch.nn.functional as F
from inception_modules import ModifiedInceptionModule
from dataset_utils.config import Alzheimer_CFG

config = Alzheimer_CFG()

class Attention(nn.Module):

    def __init__(self, dim, heads=12, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=1)
        attn = self.attn_drop(attn)

        x = (attn @ v).tranpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        def forward(self, x):
            attn_output, attn_weights = self.attn(self.norm1(x))
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights

class VanillaViT_with_ModifiedInceptionModule(nn.Module):
    '''VanillaViT with Modified Inception Module backbone
    '''

    def __init__(self, num_classes=config.num_classes, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(VanillaViT_with_ModifiedInceptionModule, self).__init__()
        self.inception = ModifiedInceptionModule(in_channels=3)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_proj = nn.Linear(64 * 4 * 32 * 32, dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
    
    def forward(self, x):
        x = self.inception(x)
        x = self.flatten(x)
        x = self.linear_proj(x)
        x = x.unsqueeze(1)

        self.attention_weights = []

        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            self.attention_weights.append(attn_weights)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x

    def get_attention_weigths(self):
        return self.attn_weights
