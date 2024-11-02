import torch
import torch.nn as nn
import math

def init_vit_weights(module, name='', gain=1.0):
    """Initialize ViT weights with Xavier/Scaled initialization"""
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # Special scaled init for QKV projections with gain
            val = math.sqrt(6. * gain) / math.sqrt(module.weight.shape[0] // 3 + module.weight.shape[1])
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 * gain / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias) 