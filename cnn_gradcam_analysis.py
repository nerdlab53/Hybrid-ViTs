import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F

# Import CNN model classes
from models.resnet import ResNet50_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer

class CNNGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get the last convolutional layer
        target_layer = self._get_last_conv_layer(model)
        if target_layer is not None:
            print(f"Target layer found: {target_layer.__class__.__name__}")
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
    
    def _get_last_conv_layer(self, model):
        """Get the last convolutional layer based on model architecture."""
        if hasattr(model, 'features'):
            # For VGG, DenseNet
            modules = list(model.features.modules())
        elif hasattr(model, 'layer4'):
            # For ResNet
            modules = list(model.layer4.modules())
        elif hasattr(model, 'blocks'):
            # For EfficientNet
            modules = list(model.blocks.modules())
        else:
            return None
            
        conv_layers = [m for m in modules if isinstance(m, torch.nn.Conv2d)]
        return conv_layers[-1] if conv_layers else None

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(input_image)
            
            if target_class is None:
                target_class = torch.argmax(output)
            
            self.model.zero_grad()
            output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            print("Warning: No gradients or activations found")
            return np.zeros((input_image.shape[2], input_image.shape[3]))

        # Calculate attention weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], 
                          mode='bilinear', align_corners=False)
        
        if cam.sum() == 0:
            print("Warning: CAM is all zeros")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
            
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()

def analyze_cnn_models(models_dir, image_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Reference the load_image function from gradcam.py
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
            # Reference the load_model_weights function from gradcam.py
            model = load_model_weights(model, checkpoint_path, device)
            if model is None:
                continue
        else:
            print(f"Warning: No weights found at {checkpoint_path}")
            continue
        
        model.eval()
        
        try:
            gradcam = CNNGradCAM(model)
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

def visualize_results(original_image, cam, save_path):
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
    
    # Create visualization
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