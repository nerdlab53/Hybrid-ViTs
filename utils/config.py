class VanillaViT:
    def __init__(self):
        self.img_size = 224
        self.num_channels = 3
        self.patch_size = 32
        self.embeddingdim = 384
        self.dropout = 0.1
        self.mlp_size = 1536
        self.num_transformer_layer = 6
        self.num_heads = 6
        self.num_classes = 4

class TinyViT:
    def __init__(self):
        self.img_size = 224
        self.num_channels = 3
        self.patch_size = 32
        self.embeddingdim = 192  # Reduced from 384
        self.dropout = 0.1
        self.mlp_size = 768  # Reduced from 1536
        self.num_transformer_layer = 3  # Reduced from 6
        self.num_heads = 3  # Reduced from 6
        self.num_classes = 4