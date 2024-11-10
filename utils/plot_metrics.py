import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

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
        'tiny_vit_deit_with_inception_v1': 'TinyViT_DeiT_with_Inception_v1',
        'tiny_vit_deit_with_modified_inception_v1': 'TinyViT_DeiT_with_ModifiedInception_v1',
        'tinydeit_inception': 'TinyViT_DeiT_with_Inception',
        'tinydeit_modifiedinception': 'TinyViT_DeiT_with_ModifiedInception'
    }
    
    print("\nScanning directories:")
    for model_dir in sorted(Path(weights_dir).glob("*weights")):
        if model_dir.is_dir():
            print(f"Found directory: {model_dir.name}")
            
            # Extract model name from directory
            model_name = model_dir.name.split('.')[-1].split('_weights')[0].lower()
            print(f"Extracted model name: {model_name}")
            
            # Try exact matching first
            mapped_name = model_mappings.get(model_name)
            
            if mapped_name is None:
                # If no exact match, try finding the longest matching key
                matching_keys = [key for key in model_mappings.keys() if key in model_name]
                if matching_keys:
                    longest_key = max(matching_keys, key=len)
                    mapped_name = model_mappings[longest_key]
                    print(f"Matched using longest key {longest_key} -> {mapped_name}")
                else:
                    mapped_name = model_name.title()
                    print(f"No mapping found, using: {mapped_name}")
            else:
                print(f"Exact match found: {mapped_name}")
            
            print(f'Loading metrics for {mapped_name}')
            metrics_file = model_dir / "metrics" / f"{mapped_name}_metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data[mapped_name] = json.load(f)
            else:
                print(f"Warning: No metrics file found at {metrics_file}")
    
    return metrics_data

def plot_individual_metrics(metrics_data, save_dir="plots"):
    """Plot individual training curves for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, metrics in metrics_data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = metrics['epochs']
        
        # Plot training and validation loss
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title(f'{model_name} - Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training and validation accuracy
        ax2.plot(epochs, metrics['train_acc'], 'b-', label='Train Accuracy')
        ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title(f'{model_name} - Accuracy Curves')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_metrics.png'), dpi=300)
        plt.close()

def plot_comparative_metrics(metrics_data, save_dir="plots"):
    """Plot comparative metrics across all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Define model categories with exact names
    model_categories = {
        'CNN Models': [
            'ResNet50', 'VGG16', 'DenseNet121', 'EfficientNet', 'MobileNetV2'
        ],
        'ViT Models': [
            'TinyViT_DeiT', 'TinyViT_ConvNeXt'
        ],
        'Hybrid Models': [
            'TinyViT_DeiT_with_Inception_v1',
            'TinyViT_DeiT_with_ModifiedInception_v1',
            'TinyViT_DeiT_with_Inception',
            'TinyViT_DeiT_with_ModifiedInception'
        ]
    }
    
    for category_name, model_names in model_categories.items():
        # Get models that exist in metrics_data
        models = [m for m in model_names if m in metrics_data]
        
        if not models:
            print(f"No models found for category: {category_name}")
            continue
            
        print(f"\nPlotting {category_name}:")
        print(f"Found models: {models}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'{category_name} Comparison', fontsize=16)
        
        # Use distinct colors for each model
        colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
        
        for model_name, color in zip(models, colors):
            metrics = metrics_data[model_name]
            epochs = metrics['epochs']
            
            # Training loss
            ax1.plot(epochs, metrics['train_loss'], color=color, label=model_name)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            
            # Validation loss
            ax2.plot(epochs, metrics['val_loss'], color=color, label=model_name)
            ax2.set_title('Validation Loss')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            
            # Training accuracy
            ax3.plot(epochs, metrics['train_acc'], color=color, label=model_name)
            ax3.set_title('Training Accuracy')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Accuracy')
            
            # Validation accuracy
            ax4.plot(epochs, metrics['val_acc'], color=color, label=model_name)
            ax4.set_title('Validation Accuracy')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Accuracy')
        
        # Add legends and grid
        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{category_name.lower().replace(" ", "_")}_comparison.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

def plot_final_metrics_comparison(metrics_data, save_dir="plots"):
    """Plot bar charts comparing final metrics across models."""
    models = list(metrics_data.keys())
    final_train_acc = [metrics['train_acc'][-1] for metrics in metrics_data.values()]
    final_val_acc = [metrics['val_acc'][-1] for metrics in metrics_data.values()]
    best_val_acc = [max(metrics['val_acc']) for metrics in metrics_data.values()]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width, final_train_acc, width, label='Final Train Acc')
    rects2 = ax.bar(x, final_val_acc, width, label='Final Val Acc')
    rects3 = ax.bar(x + width, best_val_acc, width, label='Best Val Acc')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_metrics_comparison.png'), dpi=300)
    plt.close()

def plot_vit_specific_metrics(metrics_data, save_dir="plots"):
    """Plot detailed metrics specifically for ViT models."""
    vit_patterns = ['tinyvit_deit', 'tinyvit_convnext']
    hybrid_patterns = ['inception']
    
    vit_models = [model for model in metrics_data.keys() 
                 if any(x in model.lower() for x in vit_patterns)
                 and not any(x in model.lower() for x in hybrid_patterns)]
    
    if not vit_models:
        print("No ViT models found in metrics data")
        return
    
    print(f"Found ViT models: {vit_models}")
    
    os.makedirs(os.path.join(save_dir, "vit_analysis"), exist_ok=True)
    
    # Plot attention-based metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Vision Transformer Models Analysis', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(vit_models)))
    
    for model_name, color in zip(vit_models, colors):
        metrics = metrics_data[model_name]
        epochs = metrics['epochs']
        
        # Training curves
        ax1.plot(epochs, metrics['train_loss'], color=color, linestyle='-', 
                label=f'{model_name} (Train)', linewidth=2)
        ax1.plot(epochs, metrics['val_loss'], color=color, linestyle='--', 
                label=f'{model_name} (Val)', linewidth=2)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        
        # Accuracy progression
        ax2.plot(epochs, metrics['train_acc'], color=color, linestyle='-', 
                label=f'{model_name} (Train)', linewidth=2)
        ax2.plot(epochs, metrics['val_acc'], color=color, linestyle='--', 
                label=f'{model_name} (Val)', linewidth=2)
        ax2.set_title('Accuracy Progression')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        
        # Learning rate vs Loss
        ax3.scatter(metrics['train_loss'], metrics['val_loss'], color=color, 
                   label=model_name, alpha=0.6)
        ax3.set_title('Train Loss vs Validation Loss')
        ax3.set_xlabel('Training Loss')
        ax3.set_ylabel('Validation Loss')
        
        # Accuracy distribution
        ax4.hist(metrics['val_acc'], bins=20, alpha=0.5, color=color, 
                label=model_name)
        ax4.set_title('Validation Accuracy Distribution')
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('Frequency')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vit_analysis", 'vit_detailed_metrics.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_hybrid_specific_metrics(metrics_data, save_dir="plots"):
    """Plot detailed metrics specifically for Hybrid models."""
    hybrid_models = [model for model in metrics_data.keys() 
                    if any(x in model.lower() for x in ['inception', 'modified'])]
    
    if not hybrid_models:
        print("No hybrid models found in metrics data")
        return
    
    print(f"Found hybrid models: {hybrid_models}")
    
    os.makedirs(os.path.join(save_dir, "hybrid_analysis"), exist_ok=True)
    
    # Create multiple plots for hybrid model analysis
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    fig.suptitle('Hybrid Models Analysis', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(hybrid_models)))
    
    for model_name, color in zip(hybrid_models, colors):
        metrics = metrics_data[model_name]
        epochs = metrics['epochs']
        
        # Plot metrics
        ax1.plot(epochs, metrics['val_acc'], color=color, 
                label=f'{model_name} (Val)', linewidth=2)
        ax1.plot(epochs, metrics['train_acc'], color=color, linestyle='--', 
                label=f'{model_name} (Train)', alpha=0.5)
        
        ax2.plot(epochs, metrics['train_loss'], color=color, 
                label=model_name, linewidth=2)
        
        ax3.plot(epochs, metrics['val_acc'], color=color, 
                label=model_name, linewidth=2)
        
        ax4.scatter(metrics['train_loss'], metrics['train_acc'], 
                   color=color, alpha=0.5, label=model_name)
        
        ax5.plot(epochs, 
                np.abs(np.array(metrics['train_acc']) - np.array(metrics['val_acc'])), 
                color=color, label=model_name, linewidth=2)
    
    # Set titles and labels
    ax1.set_title('Model Convergence')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    
    ax2.set_title('Training Loss Trajectory')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    
    ax3.set_title('Validation Accuracy Progression')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    
    ax4.set_title('Accuracy vs Loss Correlation')
    ax4.set_xlabel('Training Loss')
    ax4.set_ylabel('Training Accuracy')
    
    ax5.set_title('Train-Val Accuracy Gap (Stability)')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('|Train Acc - Val Acc|')
    
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hybrid_analysis", 'hybrid_detailed_metrics.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_convergence_analysis(metrics_data, save_dir="plots"):
    """Plot convergence analysis comparing all model types."""
    # Create the convergence_analysis directory
    os.makedirs(os.path.join(save_dir, "convergence_analysis"), exist_ok=True)
    
    # Update model grouping patterns
    model_groups = {
        'CNN': ['ResNet50', 'VGG16', 'DenseNet121', 'EfficientNet', 'MobileNetV2'],
        'ViT': ['TinyViT_DeiT', 'TinyViT_ConvNeXt'],
        'Hybrid': [
            'TinyViT_DeiT_with_Inception_v1',
            'TinyViT_DeiT_with_ModifiedInception_v1',
            'TinyViT_DeiT_with_Inception',
            'TinyViT_DeiT_with_ModifiedInception'
        ]
    }
    
    # Update the model grouping logic
    grouped_models = {
        group: [m for m in metrics_data.keys() if m in models]
        for group, models in model_groups.items()
    }
    
    print("\nModel grouping for convergence analysis:")
    for group, models in grouped_models.items():
        print(f"{group} models: {models}")
    
    # Create convergence analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Model Convergence Analysis', fontsize=16)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(metrics_data)))
    color_idx = 0
    
    for group_name, models in grouped_models.items():
        for model_name in models:
            metrics = metrics_data[model_name]
            epochs = metrics['epochs']
            color = colors[color_idx]
            
            # Early convergence (first 5 epochs)
            axes[0, 0].plot(epochs[:5], metrics['val_acc'][:5], 
                          label=f'{model_name}', color=color)
            
            # Overall convergence
            axes[0, 1].plot(epochs, metrics['val_acc'], 
                          label=f'{model_name}', color=color)
            
            # Stability (train-val gap)
            axes[1, 0].plot(epochs, 
                          np.abs(np.array(metrics['train_acc']) - np.array(metrics['val_acc'])), 
                          label=f'{model_name}', color=color)
            
            # Final performance scatter
            axes[1, 1].scatter(metrics['train_acc'][-1], metrics['val_acc'][-1], 
                             label=f'{model_name}', color=color, s=100)
            
            color_idx += 1
    
    axes[0, 0].set_title('Early Convergence (First 5 Epochs)')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Validation Accuracy')
    
    axes[0, 1].set_title('Overall Convergence')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Validation Accuracy')
    
    axes[1, 0].set_title('Model Stability')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('|Train Acc - Val Acc|')
    
    axes[1, 1].set_title('Final Performance')
    axes[1, 1].set_xlabel('Training Accuracy')
    axes[1, 1].set_ylabel('Validation Accuracy')
    
    for ax in axes.flat:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_analysis", 'convergence_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_final_comparison_by_category(metrics_data, save_dir="plots"):
    """Plot final metrics comparison grouped by model category."""
    categories = {
        'CNN': ['ResNet50', 'VGG16', 'DenseNet121', 'EfficientNet', 'MobileNetV2'],
        'ViT': ['TinyViT_DeiT', 'TinyViT_ConvNeXt'],
        'Hybrid': [
            'TinyViT_DeiT_with_Inception_v1',
            'TinyViT_DeiT_with_ModifiedInception_v1',
            'TinyViT_DeiT_with_Inception',
            'TinyViT_DeiT_with_ModifiedInception'
        ]
    }
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x_positions = []
    x_labels = []
    bar_width = 0.25
    spacing = 0.5
    current_x = 0
    
    for category, models in categories.items():
        # Simplified matching using exact names
        category_models = [m for m in metrics_data.keys() if m in models]
        print(f"\n{category} models found: {category_models}")
        
        for i, model in enumerate(category_models):
            metrics = metrics_data[model]
            x_pos = current_x + i * bar_width
            
            ax.bar(x_pos, metrics['val_acc'][-1], bar_width, 
                  label=f'{model} (Final)', alpha=0.8)
            ax.bar(x_pos + bar_width, max(metrics['val_acc']), bar_width,
                  label=f'{model} (Best)', alpha=0.8)
            
            x_positions.append(x_pos + bar_width/2)
            x_labels.append(model)
        
        current_x += (len(category_models) + 1) * bar_width + spacing
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_comparison_by_category.png'), dpi=300)
    plt.close()

def plot_hybrid_version_comparison(metrics_data, save_dir="plots"):
    """Plot comparison between old and new versions of hybrid models."""
    os.makedirs(os.path.join(save_dir, "hybrid_version_comparison"), exist_ok=True)
    
    hybrid_models = {
        'Inception': [
            'TinyViT_DeiT_with_Inception_v1',
            'TinyViT_DeiT_with_Inception'
        ],
        'ModifiedInception': [
            'TinyViT_DeiT_with_ModifiedInception_v1',
            'TinyViT_DeiT_with_ModifiedInception'
        ]
    }
    
    for model_type, versions in hybrid_models.items():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'{model_type} Version Comparison', fontsize=16)
        
        colors = ['blue', 'red']
        for version, color in zip(versions, colors):
            if version not in metrics_data:
                continue
                
            metrics = metrics_data[version]
            epochs = metrics['epochs']
            
            # Training Loss
            ax1.plot(epochs, metrics['train_loss'], color=color, 
                    label=f'{version} (Train)', linestyle='-')
            ax1.plot(epochs, metrics['val_loss'], color=color, 
                    label=f'{version} (Val)', linestyle='--')
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            
            # Accuracy
            ax2.plot(epochs, metrics['train_acc'], color=color, 
                    label=f'{version} (Train)', linestyle='-')
            ax2.plot(epochs, metrics['val_acc'], color=color, 
                    label=f'{version} (Val)', linestyle='--')
            ax2.set_title('Accuracy Curves')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            
            # Train-Val Accuracy Gap
            ax3.plot(epochs, 
                    np.abs(np.array(metrics['train_acc']) - np.array(metrics['val_acc'])), 
                    color=color, label=version)
            ax3.set_title('Train-Val Accuracy Gap')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('|Train Acc - Val Acc|')
            
            # Final Performance
            ax4.scatter(metrics['train_acc'][-1], metrics['val_acc'][-1], 
                       color=color, label=version, s=200)
            
        ax4.set_title('Final Performance')
        ax4.set_xlabel('Final Training Accuracy')
        ax4.set_ylabel('Final Validation Accuracy')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "hybrid_version_comparison", 
                                f'{model_type.lower()}_version_comparison.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_training_metrics(metrics_dir, selected_models=None, save_dir="plots"):
    """Main function to generate all plots."""
    try:
        metrics_data = load_metrics(metrics_dir)
        
        if not metrics_data:
            print("No metrics data found")
            return
            
        if selected_models:
            metrics_data = {k: v for k, v in metrics_data.items() if k in selected_models}
            if not metrics_data:
                print("No matching models found in selected_models")
                return
        
        # Create save directory if specified
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nFound {len(metrics_data)} models: {list(metrics_data.keys())}")
        
        # Generate all plots
        plot_individual_metrics(metrics_data, save_dir)
        plot_comparative_metrics(metrics_data, save_dir)
        plot_final_metrics_comparison(metrics_data, save_dir)
        plot_vit_specific_metrics(metrics_data, save_dir)
        plot_hybrid_specific_metrics(metrics_data, save_dir)
        plot_model_convergence_analysis(metrics_data, save_dir)
        plot_final_comparison_by_category(metrics_data, save_dir)
        plot_hybrid_version_comparison(metrics_data, save_dir)
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics for all models')
    parser.add_argument('--metrics_dir', type=str, required=True,
                      help='Directory containing model weight folders')
    parser.add_argument('--models', nargs='+', default=None,
                      help='List of specific models to plot (optional)')
    parser.add_argument('--save_dir', type=str, default='plots',
                      help='Directory to save plots (default: plots)')
    args = parser.parse_args()
    
    print("Loading metrics...")
    plot_training_metrics(args.metrics_dir, args.models, args.save_dir)
    
    print(f"\nAll plots have been saved to: {args.save_dir}")

if __name__ == "__main__":
    main()