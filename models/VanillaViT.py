import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.config import VanillaViT
from utils.patch_embedding import PatchEmbedding

config = VanillaViT()

class Attention(nn.Module):
    def __init__(self, dim, heads=12, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
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

class VanillaViT(nn.Module):
    '''VanillaViT Implementation
    '''
    def __init__(
        self, 
        img_size: int = config.img_size,
        num_channels: int = config.num_channels,
        patch_size=config.patch_size,
        embeddingdim=config.embeddingdim,
        dropout=config.dropout,
        mlp_size=config.mlp_size,
        num_transformer_layer=config.num_transformer_layer,
        num_heads = config.num_heads,
        num_classes=config.num_classes
    ):
        super().__init__()
        
        assert img_size % patch_size == 0, f'Image size is indivisible by patch size'

        self.patch_embedding = PatchEmbedding(
            in_channels=num_channels,
            patch_size=patch_size,
            embeddingdim=embeddingdim
        )

        self.class_token = nn.Parameter(
            torch.randn(1, 1, embeddingdim),
            requires_grad=True
        )

        num_patches = (img_size * img_size) // patch_size**2

        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches+1, embeddingdim)
        )

        self.embedding_dropout = nn.Dropout(p=dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embeddingdim, num_heads, mlp_size, dropout) 
            for _ in range(num_transformer_layer)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embeddingdim),
            nn.Linear(embeddingdim, num_classes)
        )

        self.attention_weights = []

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)

        self.attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            self.attention_weights.append(attn_weights)

        x = self.mlp_head(x[:, 0])
        return x

    def get_attention_weights(self):
        return self.attention_weights