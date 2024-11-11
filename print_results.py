import json
import argparse
from pathlib import Path
import numpy as np

def print_model_results(model_name, metrics):
    """Print detailed results for a single model"""
    print("\n" + "="*80)
    print(f"Model: {model_name}")
    print("="*80)
    
    # Basic metrics
    print(f"\nAccuracy Metrics:")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    
    # Per-class metrics
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    print("\nPer-class Performance:")
    print("-"*40)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*40)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['precision_per_class'][i]
        recall = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        print(f"{class_name:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-"*40)
    conf_matrix = np.array(metrics['confusion_matrix'])
    
    # Print header
    print(f"{'Actual/Predicted':<15}", end='')
    for class_name in class_names:
        print(f"{class_name[:7]:>8}", end='')
    print()
    
    # Print matrix with class names
    for i, class_name in enumerate(class_names):
        print(f"{class_name[:15]:<15}", end='')
        for j in range(len(class_names)):
            print(f"{conf_matrix[i][j]:>8}", end='')
        print()

def main():
    parser = argparse.ArgumentParser(description='Print detailed evaluation results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to the evaluation results JSON file')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to print results for')
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Print summary statistics
    print("\nSummary Statistics")
    print("="*80)
    print(f"Total models evaluated: {len(results)}")
    
    # Find best performing model
    best_model = max(results.items(), key=lambda x: x[1]['top1_accuracy'])
    print(f"Best performing model: {best_model[0]} "
          f"(Top-1 Accuracy: {best_model[1]['top1_accuracy']:.2f}%)")
    
    # Print detailed results
    if args.model:
        if args.model in results:
            print_model_results(args.model, results[args.model])
        else:
            print(f"\nError: Model '{args.model}' not found in results")
    else:
        # Print all models
        for model_name, metrics in results.items():
            print_model_results(model_name, metrics)

if __name__ == "__main__":
    main()