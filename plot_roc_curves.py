import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_tpr_fpr_from_confusion_matrix(confusion_matrix, class_idx):
    """Calculate TPR and FPR for a specific class from confusion matrix."""
    # For binary classification for each class (one-vs-rest)
    TP = confusion_matrix[class_idx][class_idx]
    FP = sum(confusion_matrix[i][class_idx] for i in range(len(confusion_matrix))) - TP
    FN = sum(confusion_matrix[class_idx]) - TP
    TN = sum(sum(row) for row in confusion_matrix) - (TP + FP + FN)
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    return TPR, FPR

def calculate_auc(fpr_points, tpr_points):
    """Calculate AUC using trapezoidal rule"""
    return np.trapz(tpr_points, fpr_points)

def plot_roc_curves(results_file, save_dir="evaluation_plots"):
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    # Plot ROC curves for each model
    plt.figure(figsize=(12, 8))
    
    # Use different colors for each model
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(results)))
    
    for model_idx, (model_name, metrics) in enumerate(results.items()):
        # Calculate micro-average ROC curve
        tprs = []
        fprs = []
        
        confusion_matrix = np.array(metrics['confusion_matrix'])
        
        # Calculate ROC curve for each class
        for class_idx in range(len(class_names)):
            tpr, fpr = calculate_tpr_fpr_from_confusion_matrix(confusion_matrix, class_idx)
            tprs.append(tpr)
            fprs.append(fpr)
        
        # Calculate average points for the curve
        mean_tpr = np.mean(tprs)
        mean_fpr = np.mean(fprs)
        
        # Calculate AUC using three points: (0,0), (mean_fpr,mean_tpr), (1,1)
        fpr_points = [0, mean_fpr, 1]
        tpr_points = [0, mean_tpr, 1]
        roc_auc = calculate_auc(fpr_points, tpr_points)
        
        # Plot ROC curve
        plt.plot(fpr_points, tpr_points,
                label=f'{model_name} (AUC = {roc_auc:.3f})',
                color=colors[model_idx], linestyle='-', linewidth=2)

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot per-class ROC curves for each model
    for model_name, metrics in results.items():
        plt.figure(figsize=(12, 8))
        confusion_matrix = np.array(metrics['confusion_matrix'])
        
        for class_idx, class_name in enumerate(class_names):
            tpr, fpr = calculate_tpr_fpr_from_confusion_matrix(confusion_matrix, class_idx)
            fpr_points = [0, fpr, 1]
            tpr_points = [0, tpr, 1]
            roc_auc = calculate_auc(fpr_points, tpr_points)
            
            plt.plot(fpr_points, tpr_points,
                    label=f'{class_name} (AUC = {roc_auc:.3f})',
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves per Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'{model_name}_roc_curves.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot ROC curves from evaluation results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to evaluation_results.json file')
    parser.add_argument('--save_dir', type=str, default='evaluation_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    plot_roc_curves(args.results_file, args.save_dir) 