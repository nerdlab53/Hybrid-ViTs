class VanillaViT: 
        img_size: int = 224,
        num_channels: int = 3,
        patch_size=16,
        embeddingdim=768,
        dropout=0.1,
        mlp_size=3072,
        num_transformer_layer=12,
        num_heads = 12,
        num_classes=4
