import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F

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
        """Get the target layer based on model architecture."""
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
        elif hasattr(model, 'backbone'):  # For ViT models
            if hasattr(model.backbone, 'blocks'):
                return model.backbone.blocks[-1].norm1
            elif hasattr(model.backbone, 'layers'):
                return model.backbone.layers[-1].blocks[-1].norm1
        raise ValueError(f"Unsupported model architecture: {type(model)}")
    
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

def analyze_models(models_dir, image_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    input_tensor, original_image = load_image(image_path)
    input_tensor = input_tensor.to(device)
    
    results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"\nAnalyzing {model_name}...")
        
        model = model_class()
        model = model.to(device)
        
        checkpoint_path = Path(models_dir) / model_name / "best_model.pth"
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

def create_comparative_plot(results, output_dir):
    """Create comparative analysis plot of attention patterns."""
    # Define model categories
    categories = {
        'CNN': ['resnet', 'vgg16', 'densenet121', 'efficientnet', 'mobilenetv2'],
        'ViT': ['tiny_vit_deit', 'tiny_vit_convnext'],
        'Hybrid': [
            'tiny_vit_deit_with_inception',
            'tiny_vit_deit_with_modified_inception'
        ]
    }
    
    metrics = ['mean_attention', 'max_attention', 'attention_coverage']
    
    # Create figure with subplots for each category
    fig, axes = plt.subplots(len(categories), len(metrics), figsize=(20, 15))
    fig.suptitle('Comparative Analysis of Attention Patterns by Model Type')
    
    for i, (category, models) in enumerate(categories.items()):
        category_results = {k: v for k, v in results.items() if k in models}
        
        for j, metric in enumerate(metrics):
            values = [results[model][metric] for model in models if model in results]
            model_names = [model for model in models if model in results]
            
            axes[i, j].bar(model_names, values)
            axes[i, j].set_title(f'{category} - {metric.replace("_", " ").title()}')
            axes[i, j].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_comparison_by_category.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def load_model_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model
    except Exception as e:
        print(f"Error loading weights from {checkpoint_path}: {str(e)}")
        return None

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