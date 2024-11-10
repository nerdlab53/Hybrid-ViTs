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

def load_model_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = None
        
        # Get the state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Filter out unexpected keys
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if k in model_state_dict and 
                             v.shape == model_state_dict[k].shape}
        
        # Load filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Print loading statistics
        missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        return model
    except Exception as e:
        print(f"Error loading weights from {checkpoint_path}: {str(e)}")
        return None