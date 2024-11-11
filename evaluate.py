import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm
from models.TinyViT_DeiT import TinyViT_DeiT
from models.TinyViT_ConvNeXt import TinyViT_ConvNeXt
from models.TinyViT_DeiT_with_Inception import TinyViT_DeiT_with_Inception
from models.TinyViT_DeiT_with_ModifiedInception import TinyViT_DeiT_with_ModifiedInception
from models.resnet import ResNet50_for_Alzheimer
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer

class ModelEvaluator:
    def __init__(self, models_dir, data_loader, device, output_dir):
        self.models_dir = Path(models_dir)
        self.data_loader = data_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models dictionary
        self.models = {
            "TinyViT_DeiT": TinyViT_DeiT(),
            "TinyViT_ConvNeXt": TinyViT_ConvNeXt(),
            "TinyViT_DeiT_with_Inception": TinyViT_DeiT_with_Inception(),
            "TinyViT_DeiT_with_ModifiedInception": TinyViT_DeiT_with_ModifiedInception(),
            "ResNet50": ResNet50_for_Alzheimer(),
            "DenseNet121": DenseNet_for_Alzheimer(),
            "EfficientNet-B0": EfficientNet_for_Alzheimer(),
            "VGG16": VGG_for_Alzheimer(),
            "MobileNetV2": MobileNet_for_Alzheimer()
        }

    def load_model(self, model, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = None
            
            # Get the state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Filter out unexpected keys and handle size mismatches
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Skipping parameter {k} due to shape mismatch: "
                              f"checkpoint: {v.shape} vs model: {model_state_dict[k].shape}")
            
            # Load filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def evaluate_single_model(self, model, model_name):
        model.eval()
        total = 0
        metrics = {
            'top1_correct': 0,
            'top5_correct': 0,
            'predictions': [],
            'true_labels': []
        }
        
        with torch.no_grad():
            for images, labels in tqdm(self.data_loader, desc=f"Evaluating {model_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                
                # Top-1 accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                metrics['top1_correct'] += (predicted == labels).sum().item()
                
                # Top-k accuracy (k = min(5, num_classes))
                num_classes = outputs.size(1)  # Get number of classes from output size
                k = min(5, num_classes)  # Use either 5 or num_classes, whichever is smaller
                _, topk_preds = outputs.topk(k, 1, True, True)
                metrics['top5_correct'] += (topk_preds == labels.view(-1, 1)).any(dim=1).sum().item()
                
                # Store predictions and labels for later analysis
                metrics['predictions'].extend(predicted.cpu().numpy())
                metrics['true_labels'].extend(labels.cpu().numpy())
        
        # Calculate final metrics
        results = {
            'model_name': model_name,
            'top1_accuracy': (metrics['top1_correct'] / total) * 100,
            'top5_accuracy': (metrics['top5_correct'] / total) * 100,
            'num_classes': num_classes
        }
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            metrics['true_labels'],
            metrics['predictions'],
            average=None
        )
        
        results.update({
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'confusion_matrix': confusion_matrix(
                metrics['true_labels'],
                metrics['predictions']
            ).tolist()
        })
        
        return results

    def evaluate_all_models(self):
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}")
            
            # Map model names to directory names
            dir_name_map = {
                'ResNet50': 'resnet',
                'VGG16': 'vgg16',
                'DenseNet121': 'densenet121',
                'EfficientNet-B0': 'efficientnet',
                'MobileNetV2': 'mobilenetv2',
                'TinyViT_DeiT': 'tiny_vit_deit',
                'TinyViT_ConvNeXt': 'tiny_vit_convnext',
                'TinyViT_DeiT_with_Inception': 'tiny_vit_deit_with_inception',
                'TinyViT_DeiT_with_ModifiedInception': 'tiny_vit_deit_with_modified_inception'
            }
            
            # Get the correct directory name
            dir_name = dir_name_map.get(model_name, model_name.lower())
            
            # Check for both possible checkpoint names
            checkpoint_paths = [
                self.models_dir / dir_name / "checkpoint_best.pth",
                self.models_dir / dir_name / "best.pth",
                self.models_dir / f"{dir_name}_weights" / "checkpoint_best.pth"
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if path.exists():
                    checkpoint_path = path
                    break
                    
            if checkpoint_path is None:
                print(f"No checkpoint found for {model_name} in paths:")
                for path in checkpoint_paths:
                    print(f"- {path}")
                continue
                
            print(f"Found checkpoint at: {checkpoint_path}")
            
            model = self.load_model(model, checkpoint_path)
            if model is None:
                continue
                
            model = model.to(self.device)
            results = self.evaluate_single_model(model, model_name)
            all_results[model_name] = results
            
            # Print immediate results
            print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
            print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
            
        # Save all results
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return all_results

def main():
    import argparse
    from utils.data_loader import load_alzheimers_data
    
    parser = argparse.ArgumentParser(description='Evaluate all models on the test set')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--dataset_type', type=str, choices=['Original', 'Augmented'], 
                       default='Original', help='Which dataset type to use')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data with validation set as test set since we don't have a separate test set
    train_loader, val_loader, _ = load_alzheimers_data(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        dataset_type=args.dataset_type
    )
    
    if val_loader is None:
        raise ValueError("Failed to load validation/test data loader")
    
    # Initialize evaluator and run evaluation
    evaluator = ModelEvaluator(args.models_dir, val_loader, device, args.output_dir)
    results = evaluator.evaluate_all_models()
    
    # Print final summary
    print("\nEvaluation Complete! Summary of Results:")
    print("=" * 80)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print("\nPer-class Precision:")
        for i, p in enumerate(metrics['precision_per_class']):
            print(f"Class {i}: {p:.4f}")

if __name__ == "__main__":
    main()