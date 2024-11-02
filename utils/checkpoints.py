import os
import torch

def save_checkpoint(state, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the latest checkpoint
    filename = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(state, filename)
    
    # If this is the best model so far, save it as the best model
    if is_best:
        best_filename = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(state, best_filename)
        
def load_checkpoint(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint['best_val_acc']
    return 0.0 