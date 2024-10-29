import matplotlib.pyplot as plt
import json
import os

def plot_training_metrics(metrics_dir, models=None, save_dir=None):
    """Plot training metrics for one or more models.
    
    Args:
        metrics_dir: Directory containing the metrics files
        models: List of model names to plot. If None, plot all available models.
        save_dir: Directory to save plots. If None, show plots instead.
    """
    if models is None:
        # Get all JSON files in metrics directory
        models = [f.split('_metrics.json')[0] for f in os.listdir(metrics_dir) 
                 if f.endswith('_metrics.json')]
    
    # Set up the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for model in models:
        # Load metrics
        with open(os.path.join(metrics_dir, f'{model}_metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        epochs = metrics['epochs']
        
        # Plot training loss
        ax1.plot(epochs, metrics['train_loss'], label=f'{model} (train)')
        ax1.plot(epochs, metrics['val_loss'], label=f'{model} (val)')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, metrics['train_acc'], label=f'{model} (train)')
        ax2.plot(epochs, metrics['val_acc'], label=f'{model} (val)')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot train vs val loss
        ax3.scatter(metrics['train_loss'], metrics['val_loss'], label=model)
        ax3.set_title('Train vs Validation Loss')
        ax3.set_xlabel('Train Loss')
        ax3.set_ylabel('Validation Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Plot train vs val accuracy
        ax4.scatter(metrics['train_acc'], metrics['val_acc'], label=model)
        ax4.set_title('Train vs Validation Accuracy')
        ax4.set_xlabel('Train Accuracy')
        ax4.set_ylabel('Validation Accuracy')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()
    else:
        plt.show() 