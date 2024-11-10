import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse

def load_metrics(weights_dir):
    """Load metrics from all model directories."""
    metrics_data = {}
    
    for model_dir in Path(weights_dir).glob("*"):
        if model_dir.is_dir() and "weights" in model_dir.name:
            model_name = model_dir.name.split('.')[1].split('_weights')[0]
            metrics_file = model_dir / "metrics" / f"{model_name}_metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data[model_name] = json.load(f)
    
    return metrics_data

def plot_individual_metrics(metrics_data, save_dir="plots"):
    """Plot individual training curves for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, metrics in metrics_data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = metrics['epochs']
        
        # Plot training and validation loss
        ax1.plot(epochs, metrics['train_loss'], label='Train Loss')
        ax1.plot(epochs, metrics['val_loss'], label='Validation Loss')
        ax1.set_title(f'{model_name} - Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training and validation accuracy
        ax2.plot(epochs, metrics['train_acc'], label='Train Accuracy')
        ax2.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    colors = sns.color_palette("husl", len(metrics_data))
    
    for (model_name, metrics), color in zip(metrics_data.items(), colors):
        epochs = metrics['epochs']
        
        # Training loss
        ax1.plot(epochs, metrics['train_loss'], label=model_name, color=color)
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        
        # Validation loss
        ax2.plot(epochs, metrics['val_loss'], label=model_name, color=color)
        ax2.set_title('Validation Loss Comparison')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        
        # Training accuracy
        ax3.plot(epochs, metrics['train_acc'], label=model_name, color=color)
        ax3.set_title('Training Accuracy Comparison')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        
        # Validation accuracy
        ax4.plot(epochs, metrics['val_acc'], label=model_name, color=color)
        ax4.set_title('Validation Accuracy Comparison')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
    
    # Add legends and grid
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparative_metrics.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_final_metrics_comparison(metrics_data, save_dir="plots"):
    """Plot bar charts comparing final metrics across models."""
    final_metrics = {
        'Model': [],
        'Final Train Acc': [],
        'Final Val Acc': [],
        'Best Val Acc': []
    }
    
    for model_name, metrics in metrics_data.items():
        final_metrics['Model'].append(model_name)
        final_metrics['Final Train Acc'].append(metrics['train_acc'][-1])
        final_metrics['Final Val Acc'].append(metrics['val_acc'][-1])
        final_metrics['Best Val Acc'].append(max(metrics['val_acc']))
    
    df = pd.DataFrame(final_metrics)
    
    plt.figure(figsize=(12, 6))
    x = range(len(df['Model']))
    width = 0.25
    
    plt.bar(x, df['Final Train Acc'], width, label='Final Train Acc')
    plt.bar([i + width for i in x], df['Final Val Acc'], width, label='Final Val Acc')
    plt.bar([i + 2*width for i in x], df['Best Val Acc'], width, label='Best Val Acc')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Final Metrics Comparison')
    plt.xticks([i + width for i in x], df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_metrics_comparison.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, default='.',
                      help='Directory containing model weight folders')
    parser.add_argument('--save_dir', type=str, default='visualization_plots',
                      help='Directory to save plots')
    args = parser.parse_args()
    
    metrics_data = load_metrics(args.weights_dir)
    
    plot_individual_metrics(metrics_data, args.save_dir)
    plot_comparative_metrics(metrics_data, args.save_dir)
    plot_final_metrics_comparison(metrics_data, args.save_dir)

if __name__ == "__main__":
    main()