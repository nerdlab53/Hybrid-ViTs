import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np

class GradientAccumulationWrapper:
    def __init__(self, model, accumulation_steps=4, max_grad_norm=1.0):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        
    def __call__(self, optimizer, scaler, loss):
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass with scaled loss
        scaler.scale(scaled_loss).backward()
        
        self.current_step += 1
        
        if self.current_step % self.accumulation_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
            
            # Optimizer step and scaler update
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

class CyclicLRWithRestarts:
    def __init__(self, optimizer, total_epochs, cycles=3, cycle_mult=2.0):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.cycles = cycles
        self.cycle_mult = cycle_mult
        self.current_epoch = 0
        
        # Calculate cycle lengths
        self.cycle_lengths = []
        cycle_length = total_epochs // cycles
        for i in range(cycles):
            self.cycle_lengths.append(cycle_length)
            cycle_length = int(cycle_length * cycle_mult)
            
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        lr_mult = self.get_lr()
        
        # Update learning rate for all param groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_mult
    
    def get_lr(self):
        """Calculate the learning rate multiplier"""
        # Find current cycle
        current_cycle = 0
        epoch_in_cycle = self.current_epoch
        for length in self.cycle_lengths:
            if epoch_in_cycle < length:
                break
            epoch_in_cycle -= length
            current_cycle += 1
            
        # If we've exceeded all cycles, use the minimum learning rate
        if current_cycle >= len(self.cycle_lengths):
            return 0.1  # minimum learning rate multiplier
            
        # Calculate position in cycle (0 to 1)
        pos = epoch_in_cycle / self.cycle_lengths[current_cycle]
        
        # Cosine annealing with warm restarts
        return 0.5 * (1 + np.cos(np.pi * pos))
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        return {
            'current_epoch': self.current_epoch,
            'cycle_lengths': self.cycle_lengths,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state."""
        self.current_epoch = state_dict['current_epoch']
        self.cycle_lengths = state_dict['cycle_lengths']