import torch 
import torch.nn as nn
import torch.nn.functional as F
from inception_modules import ModifiedInceptionModule
from dataset_utils.config import Alzheimer_CFG

config = Alzheimer_CFG()

class VanillaViT_with_ModifiedInceptionModule(nn.Module):
    '''VanillaViT with Modified Inception Module backbone
    '''

    def __init__(self, num_classes=config.num_classes, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(VanillaViT_with_ModifiedInceptionModule, self).__init__()
        self.inception = ModifiedInceptionModule(in_channels=3)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_proj = nn.Linear(64 * 4 * 32 * 32, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim,
                                       nhead=heads,
                                       dim_feedforward=mlp_dim,
                                       dropout=dropout),
            num_layers=depth
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        x = self.inception(x)
        x = self.flatten(x)
        x = self.linear_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x
