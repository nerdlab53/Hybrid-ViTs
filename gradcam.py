import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F
import math

# Import all model classes
from models.resnet import ResNet50_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer
from models.TinyViT_DeiT import TinyViT_DeiT
from models.TinyViT_ConvNeXt import TinyViT_ConvNeXt
from models.TinyViT_DeiT_with_Inception import TinyViT_DeiT_with_Inception
from models.TinyViT_DeiT_with_ModifiedInception import TinyViT_DeiT_with_ModifiedInception

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output[0].detach()
        
        # Get target layer based on model architecture
        target = self._get_target_layer_custom(model)
        if target is not None:
            target.register_forward_hook(forward_hook)
            target.register_full_backward_hook(backward_hook)
    
    def _get_target_layer_custom(self, model):
        """Get target layer based on model architecture."""
        # For CNN models
        if hasattr(model, 'resnet'):
            return model.resnet.layer4[-1]
        elif hasattr(model, 'vgg'):
            return model.vgg.features[-1]
        elif hasattr(model, 'densenet'):
            return model.densenet.features.denseblock4
        elif hasattr(model, 'efficientnet'):
            return model.efficientnet.features[-1]
        elif hasattr(model, 'mobilenet'):
            return model.mobilenet.features[-1]
        # For transformer and hybrid models
        elif hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'norm'):
                # For DeiT and hybrid models
                return model.backbone.norm
        elif hasattr(model, 'norm'):
            # For base TinyViT
            return model.norm
        return None

    def reshape_transform(self, tensor):
        """Transform tensor based on architecture."""
        if len(tensor.shape) == 3:  # [B, N, C]
            B, N, C = tensor.shape
            if N > 1:  # Has CLS token
                tensor = tensor[:, 1:]  # Remove CLS token
            
            H = W = int(math.sqrt(tensor.shape[1]))
            tensor = tensor.reshape(B, H, W, C)
            tensor = tensor.permute(0, 3, 1, 2)
            return tensor
        return tensor

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            model_output = self.model(input_image)
            
            if target_class is None:
                target_class = torch.argmax(model_output)
            
            self.model.zero_grad()
            
            one_hot = torch.zeros_like(model_output)
            one_hot[0, target_class] = 1
            model_output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("Warning: No gradients or activations found")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        # Transform activations and gradients
        activations = self.reshape_transform(self.activations)
        gradients = self.reshape_transform(self.gradients)
        
        # Calculate weights and CAM
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Post-process CAM
        cam = F.relu(cam)
        if cam.sum() == 0:
            print("Warning: CAM is all zeros")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        cam = F.interpolate(cam, size=input_image.shape[2:], 
                          mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()

def load_image(image_path, size=224):
    """Load and preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, image

def apply_gradcam(model, input_tensor, original_image, save_path):
    """Apply GradCAM and save visualization."""
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer=None)  # target_layer is determined automatically
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor.cuda())
    
    # Convert original image to numpy array
    orig_img = np.array(original_image)
    
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Combine original image with heatmap
    alpha = 0.5
    superimposed = cv2.addWeighted(orig_img, 1-alpha, heatmap, alpha, 0)
    
    # Create figure with original, heatmap, and superimposed images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(orig_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(heatmap)
    ax2.set_title('GradCAM Heatmap')
    ax2.axis('off')
    
    ax3.imshow(superimposed)
    ax3.set_title('Superimposed')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cam

def load_model_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        
        # Get the current model's state dict
        model_state_dict = model.state_dict()
        
        # Filter and adjust the loaded state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
        
        # Load filtered state dict
        model.load_state_dict(new_state_dict, strict=False)
        return model
    except Exception as e:
        print(f"Error loading weights from {checkpoint_path}: {str(e)}")
        return None

def create_comparative_plot(results, output_dir):
    """Create a comparative plot of attention metrics across models."""
    models = list(results.keys())
    metrics = ['mean_attention', 'max_attention', 'attention_coverage']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_analysis.png'))
    plt.close()

def analyze_models(models_dir, image_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    input_tensor, original_image = load_image(image_path)
    input_tensor = input_tensor.to(device)
    
    model_classes = {
        'resnet': ResNet50_for_Alzheimer,
        'vgg16': VGG_for_Alzheimer,
        'densenet121': DenseNet_for_Alzheimer,
        'efficientnet': EfficientNet_for_Alzheimer,
        'mobilenetv2': MobileNet_for_Alzheimer,
        'tiny_vit_deit': TinyViT_DeiT,
        'tiny_vit_convnext': TinyViT_ConvNeXt,
        'tiny_vit_deit_with_inception': TinyViT_DeiT_with_Inception,
        'tiny_vit_deit_with_modified_inception': TinyViT_DeiT_with_ModifiedInception
    }
    
    results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"\nAnalyzing {model_name}...")
        
        model = model_class()
        model = model.to(device)
        
        checkpoint_path = Path(models_dir) / model_name / "checkpoint_best.pth"
        if checkpoint_path.exists():
            model = load_model_weights(model, checkpoint_path, device)
            if model is None:
                continue
            print(f"Loaded weights from {checkpoint_path}")
        else:
            print(f"Warning: No weights found at {checkpoint_path}")
            continue
        
        model.eval()
        
        try:
            save_path = os.path.join(output_dir, f"{model_name}_gradcam.png")
            cam = apply_gradcam(model, input_tensor, original_image, save_path)
            
            results[model_name] = {
                'cam': cam,
                'mean_attention': float(np.mean(cam)),
                'max_attention': float(np.max(cam)),
                'attention_coverage': float(np.mean(cam > 0.5))
            }
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue
    
    create_comparative_plot(results, output_dir)
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze models using GradCAM')
    parser.add_argument('--models_dir', type=str, required=True,
                      help='Directory containing model weights')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image for analysis')
    parser.add_argument('--output_dir', type=str, default='gradcam_analysis',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    results = analyze_models(args.models_dir, args.image_path, args.output_dir)