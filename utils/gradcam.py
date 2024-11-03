import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks based on model type
        if hasattr(model, 'backbone'):  # Pretrained models
            target = self._get_target_layer_pretrained(model)
        else:  # Custom models
            target = self._get_target_layer_custom(model)
            
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)
    
    def _get_target_layer_pretrained(self, model):
        # For DeiT, Swin, and ConvNeXt
        if hasattr(model.backbone, 'blocks'):
            return model.backbone.blocks[-1].norm1  # Last transformer block
        elif hasattr(model.backbone, 'stages'):
            return model.backbone.stages[-1][-1]  # Last stage, last block
        return model.backbone.head  # Fallback
    
    def _get_target_layer_custom(self, model):
        # For custom TinyViT variants
        return model.transformer_blocks[-1].norm1
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        model_output[0, target_class].backward()
        
        # Get weights
        gradients = self.gradients.mean((2, 3), keepdim=True)
        activations = self.activations
        
        # Weight the activations by the gradients
        weights = F.adaptive_avg_pool2d(gradients, 1)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()[0, 0] 