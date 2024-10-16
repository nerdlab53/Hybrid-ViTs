# Implementation of the VanillaViT 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.config import VanillaViT

config = VanillaViT()

class PatchEmbedding(
    nn.Module
):
    def __init__(self, in_channels:int = 3, patch_size: int = 16, embeddingdim: int = 768):
        super().__init__()
        self.patch_size = 16
        self.conv_patch_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embeddingdim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten = nn.flatten(start_dim=2, end_dim=3)

    def forward(self, x):

        assert x.shape[-1] % self.patch_size == 0, f'Input Image is not divisible by patch size'

        patches = self.conv_patch_layer(x)
        flattened = self.flatten(patches)
        return flattened.permute(0, 2, 1)
        
        # After flattening, the shape comes out to be (batch, num_patches, embeddingdim)


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
        num_classes=config.num_channels
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
            torch.randn(1, 1, num_patches+1, embeddingdim)
        )

        self.embedding_dropout = nn.Dropout(p=dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model = embeddingdim,
                nhead=num_heads,
                dim_feedforward = mlp_size,
                activation = 'gelu',
                batch_first = True,
                norm_first = True,

            ),
            num_layers = num_transformer_layer
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embeddingdim),
            nn.Linear(in_features=embeddingdim,
                      out_features=num_classes)
        )

        def forward(self, x):
            batch_size = x.shape[0]
            x = self.patch_embedding(x)
            class_token = self.class_token.expand(batch_size, -1, -1)
            x = torch.cat((class_token, x), dim=1)
            x = self.positional_embedding + x
            x = self.embedding_dropout(x)
            x = self.transformer_encoder(x)
            x = self.mlp_head(x[:, 0])
            return x
