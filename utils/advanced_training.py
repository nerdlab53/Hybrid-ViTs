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
        
        # Calculate cycle lengths
        self.cycle_lengths = []
        cycle_length = total_epochs // cycles
        for i in range(cycles):
            self.cycle_lengths.append(cycle_length)
            cycle_length = int(cycle_length * cycle_mult)
            
    def get_lr(self, epoch):
        # Find current cycle
        current_cycle = 0
        epoch_in_cycle = epoch
        for length in self.cycle_lengths:
            if epoch_in_cycle < length:
                break
            epoch_in_cycle -= length
            current_cycle += 1
            
        # Calculate position in cycle (0 to 1)
        pos = epoch_in_cycle / self.cycle_lengths[current_cycle]
        
        # Cosine annealing with warm restarts
        return 0.5 * (1 + np.cos(np.pi * pos)) 