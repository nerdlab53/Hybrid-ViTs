import torch
import torch.nn as nn
import timm

class PretrainedTinyViTBase(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        img_size=224,
        num_channels=3,
        patch_size=16,
        num_classes=4,
        dropout=0.1,
        freeze_backbone=True
    ):
        super().__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            pretrained_model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=img_size,
            in_chans=num_channels
        )
        
        # Get embedding dimension from backbone
        self.embed_dim = self.backbone.embed_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add custom head with proper dimensions
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        self.attention_weights = []
    
    def forward(self, x):
        # Get features from backbone (B, N, D)
        x = self.backbone.forward_features(x)
        
        # Apply classification head
        # First ensure x is 2D for the linear layers (B, D)
        if len(x.shape) == 3:
            x = x[:, 0]  # Take CLS token
        
        x = self.mlp_head(x)
        return x