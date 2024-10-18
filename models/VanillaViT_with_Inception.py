import torch 
import torch.nn as nn
import torch.nn.functional as F
from inception_modules import InceptionModule
from dataset_utils.config import Alzheimer_CFG

config = Alzheimer_CFG()

class Attention(nn.Module):

    def __init__(self, dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.heads = num_heads
        self.scale = dim** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        B, N, C = x.shape
        #(B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        #(B, N, C) -> (B, N, 3, heads, C//heads).permute(2, 0, 3, 1, 4) -> (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #(B, heads, N, C // heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        #(B, heads, N, C // heads) -> (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #(B, heads, N, N) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # Output : (B, N, C)
        return x, attn

class TransformerBlock(nn.Module):
    
    def __init__(self, dim, heads, mlp_dim, dropout=-0.1):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
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

class VanillaViT_with_Inception(nn.Module):
    '''VanillaViT with InceptionModule backbone
    '''

    def __init__(
            self,
            num_classes=config.num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1
    ):
        super(VanillaViT_with_Inception, self).__init__()
        self.inception = InceptionModule(in_channels=3)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_proj = nn.Linear(64 * 4 * 32 * 32, dim)
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.attention_weights = []

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

    def get_attention_weights(self):
        return self.attention_weights
