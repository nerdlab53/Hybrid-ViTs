# Implementation of the VanillaViT 
import torch 
import torch.nn as nn
import torch.nn.functional as F

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


