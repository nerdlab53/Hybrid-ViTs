import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(weights_dir):
    """Load metrics from all model directories."""
    metrics_data = {}
    
    # Define exact model name mappings based on directory structure
    model_mappings = {
        'resnet': 'ResNet50',
        'vgg16': 'VGG16',
        'densenet121': 'DenseNet121',
        'efficientnet': 'EfficientNet',
        'mobilenetv2': 'MobileNetV2',
        'tiny_vit_deit': 'TinyViT_DeiT',
        'tiny_vit_convnext': 'TinyViT_ConvNeXt',
        'tinydeit_inception': 'TinyViT_DeiT_with_Inception',
        'tinydeit_modifiedinception': 'TinyViT_DeiT_with_ModifiedInception'
    }
    
    weights_dir = Path(weights_dir)
    for model_dir in weights_dir.glob("*weights"):
        if model_dir.is_dir():
            model_name = model_dir.name.split('_weights')[0]
            mapped_name = model_mappings.get(model_name, model_name)
            
            metrics_file = model_dir / "metrics" / f"{mapped_name}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data[mapped_name] = json.load(f)
    
    return metrics_data

def plot_roc_curves(metrics_data, save_dir="roc_plots"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curves for all models
    plt.figure(figsize=(12, 8))
    
    # Use different colors for each model
    colors = plt.cm.tab20(np.linspace(0, 1, len(metrics_data)))
    
    for model_idx, (model_name, metrics) in enumerate(metrics_data.items()):
        if 'val_roc' in metrics:
            fpr = metrics['val_roc']['fpr']
            tpr = metrics['val_roc']['tpr']
            auc = metrics['val_roc']['auc']
            
            plt.plot(fpr, tpr, 
                    label=f'{model_name} (AUC = {auc:.3f})',
                    color=colors[model_idx], linewidth=2)
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot individual ROC curves for each model
    for model_name, metrics in metrics_data.items():
        if 'val_roc_per_class' in metrics:
            plt.figure(figsize=(10, 8))
            
            class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            for class_idx, class_name in enumerate(class_names):
                if class_idx in metrics['val_roc_per_class']:
                    class_metrics = metrics['val_roc_per_class'][str(class_idx)]
                    fpr = class_metrics['fpr']
                    tpr = class_metrics['tpr']
                    auc = class_metrics['auc']
                    
                    plt.plot(fpr, tpr, 
                            label=f'{class_name} (AUC = {auc:.3f})',
                            linewidth=2)
            
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curves per Class')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / f'{model_name}_roc_curves.png', dpi=300)
            plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot ROC curves from metrics files')
    parser.add_argument('--weights_dir', type=str, required=True,
                       help='Directory containing model weights folders')
    parser.add_argument('--save_dir', type=str, default='roc_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    metrics_data = load_metrics(args.weights_dir)
    plot_roc_curves(metrics_data, args.save_dir) 