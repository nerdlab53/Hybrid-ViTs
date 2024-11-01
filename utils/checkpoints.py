import os
import torch

def save_checkpoint(state, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'checkpoint_best.pth')
    torch.save(state, filename)

def load_checkpoint(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint['best_val_acc']
    return 0.0 