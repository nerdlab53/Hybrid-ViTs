import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer based on model architecture
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'blocks'):
                # For DeiT models, use the last block's attention output
                target = model.backbone.blocks[-1].attn
                if target is not None:
                    print(f"Target layer found: {target.__class__.__name__}")
                    target.register_forward_hook(forward_hook)
                    target.register_full_backward_hook(backward_hook)

    def reshape_transform(self, tensor):
        if len(tensor.shape) == 3:  # [B, N, C]
            # Remove CLS token
            tensor = tensor[:, 1:]
            
            # Calculate size of feature map (sqrt of number of patches)
            n = tensor.shape[1]
            size = int(math.sqrt(n))
            
            # Reshape to [B, H, W, C]
            tensor = tensor.reshape(tensor.shape[0], size, size, -1)
            
            # Permute to [B, C, H, W]
            tensor = tensor.permute(0, 3, 1, 2)
            
            return tensor
        return tensor

    def generate_cam(self, input_image, target_class=None):
        # Enable gradients
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        input_image.requires_grad_()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Check if gradients and activations exist
        if self.gradients is None:
            print("No gradients captured!")
            return None
        if self.activations is None:
            print("No activations captured!")
            return None

        # Process gradients and activations
        gradients = self.reshape_transform(self.gradients)
        activations = self.reshape_transform(self.activations)
        
        # Weight the channels by gradient
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], 
                          mode='bilinear', align_corners=False)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Restore model state
        self.model.train(was_training)
        
        # Detach tensor before converting to numpy
        return cam.squeeze().detach().cpu().numpy()

def load_and_preprocess_image(image_path, device='cuda'):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    input_tensor = preprocess(image)
    
    # Add batch dimension and move to device
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    return input_tensor, image

def unfreeze_last_block(model):
    """Unfreeze the last transformer block for gradient computation"""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        for param in model.backbone.blocks[-1].parameters():
            param.requires_grad = True

def visualize_gradcam(image_path, model, save_path=None):
    # Prepare image
    input_tensor, original_image = load_and_preprocess_image(image_path)
    
    # Unfreeze last block
    unfreeze_last_block(model)
    
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer=None)
    
    # Generate heatmap
    heatmap = gradcam.generate_cam(input_tensor)
    
    if heatmap is None:
        print("Failed to generate heatmap!")
        return
    
    # Convert original image to numpy array
    original_image = np.array(original_image)
    
    # Resize heatmap to match original image size
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap = heatmap.resize(original_image.shape[:2], Image.BILINEAR)
    heatmap = np.array(heatmap) / 255.0
    
    # Create colored heatmap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    
    # Superimpose heatmap on original image
    superimposed = (original_image / 255.0 * 0.7 + heatmap_colored * 0.3)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot heatmap
    ax2.imshow(heatmap, cmap='jet')
    ax2.set_title('GradCAM Heatmap')
    ax2.axis('off')
    
    # Plot superimposed image
    ax3.imshow(superimposed)
    ax3.set_title('Superimposed')
    ax3.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Import all model variants
    from models.TinyViT_DeiT import TinyViT_DeiT
    from models.TinyViT_DeiT_with_Inception import TinyViT_DeiT_with_Inception
    from models.TinyViT_ConvNeXt import TinyViT_ConvNeXt
    
    # List of models to analyze
    models_to_analyze = [
        ('TinyViT_DeiT', TinyViT_DeiT()),
        ('TinyViT_DeiT_with_Inception', TinyViT_DeiT_with_Inception()),
        ('TinyViT_ConvNeXt', TinyViT_ConvNeXt())
    ]
    
    # Path to your image
    image_path = "/teamspace/studios/this_studio/augmented-alzheimer-mri-dataset/AugmentedAlzheimerDataset/MildDemented/0a0a0acd-8bd8-4b79-b724-cc5711e83bc7.jpg"
    
    # Analyze each model
    for model_name, model in models_to_analyze:
        print(f"\nAnalyzing {model_name}...")
        
        # Initialize model and move to GPU
        model = model.to('cuda')
        
        # Load weights
        checkpoint = torch.load(f'Model Weights/{model_name.lower()}/checkpoint_best.pth')
        model.load_state_dict(checkpoint['state_dict'])
        
        # Visualize GradCAM
        visualize_gradcam(image_path, model, save_path=f"gradcam_output_{model_name}.png")