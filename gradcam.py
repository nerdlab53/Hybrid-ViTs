import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F

# Import CNN model classes only
from models.resnet import ResNet50_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            print(f"Forward hook called. Output shape: {output.shape}")
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            print(f"Backward hook called. Grad output shape: {grad_output[0].shape}")
            self.gradients = grad_output[0].detach()
        
        # Get target layer based on model architecture
        target = self._get_target_layer_custom(model)
        if target is not None:
            print(f"Target layer found: {target.__class__.__name__}")
            target.register_forward_hook(forward_hook)
            target.register_full_backward_hook(backward_hook)
    
    def _get_target_layer_custom(self, model):
        """Get target layer based on CNN architecture."""
        if isinstance(model, ResNet50_for_Alzheimer):
            return model.layer4[-1].conv3
        elif isinstance(model, VGG_for_Alzheimer):
            return model.features[-1]
        elif isinstance(model, DenseNet_for_Alzheimer):
            return model.features.denseblock4.denselayer16.conv2
        elif isinstance(model, EfficientNet_for_Alzheimer):
            return model._blocks[-1]._project_conv
        elif isinstance(model, MobileNet_for_Alzheimer):
            return model.features[-1]
        return None

    def generate_cam(self, input_image, target_class=None):
        print("\nGenerating CAM...")
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            print("Forward pass...")
            output = self.model(input_image)
            print(f"Model output shape: {output.shape}")
            
            if target_class is None:
                target_class = torch.argmax(output)
            print(f"Target class: {target_class}")
            
            self.model.zero_grad()
            output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            print("Warning: No gradients or activations found")
            print(f"Gradients: {self.gradients}")
            print(f"Activations: {self.activations}")
            return np.zeros((input_image.shape[2], input_image.shape[3]))

        print("Processing gradients and activations...")
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], 
                          mode='bilinear', align_corners=False)
        
        if cam.sum() == 0:
            print("Warning: CAM is all zeros")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
            
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
    print(f"Shape of image: {image.size}")
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, image

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

def analyze_cnn_models(models_dir, image_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    input_tensor, original_image = load_image(image_path)
    input_tensor = input_tensor.to(device)
    
    cnn_models = {
        'resnet': ResNet50_for_Alzheimer,
        'vgg16': VGG_for_Alzheimer,
        'densenet121': DenseNet_for_Alzheimer,
        'efficientnet': EfficientNet_for_Alzheimer,
        'mobilenetv2': MobileNet_for_Alzheimer
    }
    
    results = {}
    
    for model_name, model_class in cnn_models.items():
        print(f"\nAnalyzing {model_name}...")
        
        model = model_class().to(device)
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
            gradcam = GradCAM(model)
            cam = gradcam.generate_cam(input_tensor)
            
            # Save visualization
            save_path = os.path.join(output_dir, f"{model_name}_gradcam.png")
            visualize_results(original_image, cam, save_path)
            
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
    
    parser = argparse.ArgumentParser(description='Analyze CNN models using GradCAM')
    parser.add_argument('--models_dir', type=str, required=True,
                      help='Directory containing model weights')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image for analysis')
    parser.add_argument('--output_dir', type=str, default='cnn_gradcam_analysis',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    results = analyze_cnn_models(args.models_dir, args.image_path, args.output_dir)