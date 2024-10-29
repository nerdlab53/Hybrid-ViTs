import argparse
from utils.plot_metrics import plot_training_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, required=True,
                       help="Directory containing metrics files")
    parser.add_argument("--models", nargs="+", default=None,
                       help="List of models to plot. If not specified, plot all models.")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save plots. If not specified, show plots instead.")
    
    args = parser.parse_args()
    plot_training_metrics(args.metrics_dir, args.models, args.save_dir)

if __name__ == "__main__":
    main() 