import torch 
import torch.nn as nn
import torch.nn.functional as F
from .inception_modules import InceptionModule
from dataset_utils.config import Alzheimer_CFG
from utils.initialization import init_vit_weights

config = Alzheimer_CFG()

class Attention(nn.Module):

    def __init__(self, dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=dropout)
        # Initialize QKV with scaled initialization
        nn.init.xavier_uniform_(self.qkv.weight)
        if hasattr(self.qkv, 'bias') and self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        
        # Initialize projection
        nn.init.xavier_uniform_(self.proj.weight)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, C = x.shape
        #(B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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
    
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        # Initialize MLP layers
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    
    def forward(self, x):
        attn_output, attn_weights = self.attn(self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class VanillaViT_with_Inception(nn.Module):
    '''VanillaViT with InceptionModule backbone
    '''

    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=4,
                 dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_dim=3072,
                 dropout=0.1):
        super().__init__()
        
        self.inception = InceptionModule(in_channels=3)
        
        # Calculate the output size after inception module
        # InceptionModule outputs 256 channels (64+64+96+32)
        inception_out_channels = 256  # Total channels after concatenation
        inception_spatial_size = img_size  # Inception maintains spatial dimensions
        
        # Add a reduction layer after inception to reduce spatial dimensions
        self.reduction = nn.Sequential(
            nn.Conv2d(inception_out_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([dim, img_size//patch_size, img_size//patch_size])
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        # Position embeddings and class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # Initialize attention weights storage
        self.attention_weights = []

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

    def forward(self, x):
        B = x.shape[0]
        
        # Clear attention weights from previous forward pass
        self.attention_weights = []
        
        # Inception feature extraction
        x = self.inception(x)  # B, 256, 224, 224
        
        # Reduce spatial dimensions
        x = self.reduction(x)  # B, dim, H/patch_size, W/patch_size
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, dim
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x, attn = block(x)
            self.attention_weights.append(attn)

        # Use [CLS] token for classification
        x = self.norm(x)
        x = x[:, 0]  # Take only the CLS token
        x = self.mlp_head(x)
        
        return x

    def get_attention_weights(self):
        return self.attention_weights
