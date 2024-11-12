import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_evaluation_results(results_file, save_dir="evaluation_plots"):
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model names and metrics
    models = list(results.keys())
    top1_accuracies = [results[m]['top1_accuracy'] for m in models]
    top5_accuracies = [results[m]['top5_accuracy'] for m in models]
    
    # 1. Accuracy Comparison Bar Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, top1_accuracies, width, label='Top-1 Accuracy')
    plt.bar(x + width/2, top5_accuracies, width, label='Top-5 Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_comparison.png')
    plt.close()
    
    # 2. Per-class Metrics Plot
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    for model in models:
        metrics = results[model]
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot precision
        ax1.bar(class_names, metrics['precision_per_class'])
        ax1.set_title('Precision per Class')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot recall
        ax2.bar(class_names, metrics['recall_per_class'])
        ax2.set_title('Recall per Class')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot F1
        ax3.bar(class_names, metrics['f1_per_class'])
        ax3.set_title('F1 Score per Class')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{model} - Per-class Metrics')
        plt.tight_layout()
        plt.savefig(save_dir / f'{model}_class_metrics.png')
        plt.close()
        
        # 3. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'{model} - Confusion Matrix')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center")
        
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.tight_layout()
        plt.savefig(save_dir / f'{model}_confusion_matrix.png')
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to evaluation_results.json file')
    parser.add_argument('--save_dir', type=str, default='evaluation_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    plot_evaluation_results(args.results_file, args.save_dir)