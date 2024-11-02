class VanillaViT:
    def __init__(self):
        self.img_size = 224
        self.num_channels = 3
        self.patch_size = 16
        self.embeddingdim = 768
        self.dropout = 0.1
        self.mlp_size = 3072
        self.num_transformer_layer = 12
        self.num_heads = 12
        self.num_classes = 4
