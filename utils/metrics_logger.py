import json
import os
import numpy as np

class MetricsLogger:
    def __init__(self, output_dir, model_name):
        self.output_dir = output_dir
        self.model_name = model_name
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['train_acc'].append(float(train_acc))
        self.metrics['val_loss'].append(float(val_loss))
        self.metrics['val_acc'].append(float(val_acc))
        
    def save(self):
        metrics_dir = os.path.join(self.output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save as JSON for easy loading
        metrics_file = os.path.join(metrics_dir, f'{self.model_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save as NPY for numerical processing
        for metric_name, values in self.metrics.items():
            np_file = os.path.join(metrics_dir, f'{self.model_name}_{metric_name}.npy')
            np.save(np_file, np.array(values)) 