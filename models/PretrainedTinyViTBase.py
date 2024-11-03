import torch
import torch.nn as nn
import timm

class PretrainedTinyViTBase(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        img_size=None,
        num_channels=3,
        patch_size=None,
        num_classes=4,
        dropout=0.1,
        freeze_backbone=True
    ):
        super().__init__()
        
        # Load pretrained model with only necessary parameters
        model_args = {
            'pretrained': True,
            'num_classes': 0,  # Remove classification head
            'in_chans': num_channels,
            'global_pool': ''  # Disable global pooling
        }
        
        # Add img_size only if needed (for ViT-like models)
        if img_size is not None:
            model_args['img_size'] = img_size
            
        self.backbone = timm.create_model(
            pretrained_model_name,
            **model_args
        )
        
        # Get embedding dimension from backbone
        if hasattr(self.backbone, 'num_features'):
            self.embed_dim = self.backbone.num_features
        else:
            self.embed_dim = self.backbone.embed_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Add custom head with proper dimensions
        self.mlp_head = nn.Sequential(
            nn.Flatten(),  # Flatten the pooled features
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        self.attention_weights = []
    
    def forward(self, x):
        # Get features from backbone
        x = self.backbone.forward_features(x)
        
        # Handle different output formats
        if len(x.shape) == 4:  # CNN-like output [B, C, H, W]
            x = self.global_pool(x)
        elif len(x.shape) == 3:  # Transformer-like output [B, N, C]
            x = x[:, 0]  # Take CLS token
            
        # Apply classification head
        x = self.mlp_head(x)
        return x