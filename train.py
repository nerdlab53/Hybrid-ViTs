import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import timedelta
# from models.VanillaViT import VanillaViT
# from models.VanillaViT_with_Inception import VanillaViT_with_Inception
# from models.VanillaViT_with_ModifiedInception import VanillaViT_with_ModifiedInceptionModule
from models.TinyViT_DeiT import TinyViT_DeiT
# from models.TinyViT_Swin import TinyViT_Swin
from models.TinyViT_ConvNeXt import TinyViT_ConvNeXt
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer
from models.resnet import ResNet50_for_Alzheimer
from utils.scheduler import WarmupCosineScheduler
from utils.checkpoints import save_checkpoint
from dataset_utils.alzheimers_dataset import AlzheimersDataset
from utils.data_loader import load_alzheimers_data
from utils.metrics_logger import MetricsLogger
from models.TinyViT import TinyViT
# from models.TinyViT_with_Inception import TinyViT_with_Inception
# from models.TinyViT_with_ModifiedInception import TinyViT_with_ModifiedInception
import copy
# from models.TinyViT_BEiT import TinyViT_BEiT
from models.TinyViT_DeiT_with_Inception import TinyViT_DeiT_with_Inception
from models.TinyViT_DeiT_with_ModifiedInception import TinyViT_DeiT_with_ModifiedInception
# from models.TinyViT_with_Inception_Advanced import TinyViT_with_Inception_Advanced
# from models.TinyViT_with_ModifiedInception_Advanced import TinyViT_with_ModifiedInception_Advanced
import math

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds==labels).mean()

def save_model(args, model, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.model_type}_{global_step}_checkpoint.pth")
    
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'global_step': global_step,
    }
    
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Set num_classes based on dataset if not specified
    if not hasattr(args, 'num_classes'):
        if args.dataset == "cifar10":
            args.num_classes = 10
        elif args.dataset == "cifar100":
            args.num_classes = 100
        elif args.dataset == "alzheimers":
            args.num_classes = 4  # Based on the Alzheimer's dataset classes
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.model_type == "VanillaViT":
        model = VanillaViT(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            embeddingdim=args.embeddingdim,
            num_heads=args.num_heads,
            mlp_size=args.mlp_size,
            num_transformer_layer=3,  # Reduced to 3
            num_classes=args.num_classes
        )
    elif args.model_type == "TinyViT":
        model = TinyViT(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            embeddingdim=args.embeddingdim,
            num_heads=args.num_heads,
            mlp_size=args.mlp_size,
            num_transformer_layer=args.num_transformer_layer,
            num_classes=args.num_classes
        )
    elif args.model_type == "VanillaViT_with_Inception":
        model = VanillaViT_with_Inception(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.num_channels,
            num_classes=args.num_classes,
            dim=192,  # Reduced dimension
            depth=3,  # Reduced to 3
            num_heads=3,
            mlp_dim=768
        )
    elif args.model_type == "TinyViT_with_Inception":
        model = TinyViT_with_Inception(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.num_channels,
            num_classes=args.num_classes,
            dim=192,
            depth=3,  # Reduced to 3
            num_heads=3,
            mlp_dim=768
        )
    elif args.model_type == "VanillaViT_with_ModifiedInception":
        model = VanillaViT_with_ModifiedInceptionModule(
            num_classes=args.num_classes,
            dim=192,  # Reduced dimension
            depth=3,  # Reduced to 3
            heads=3,
            mlp_dim=768
        )
    elif args.model_type == "TinyViT_with_ModifiedInception":
        model = TinyViT_with_ModifiedInception(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.num_channels,
            num_classes=args.num_classes,
            dim=args.embeddingdim,
            depth=args.num_transformer_layer,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_size,
            dropout=args.dropout
        )
    elif args.model_type == "TinyViT_DeiT":
        model = TinyViT_DeiT(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif args.model_type == "TinyViT_Swin":
        model = TinyViT_Swin(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif args.model_type == "TinyViT_ConvNeXt":
        model = TinyViT_ConvNeXt(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif args.model_type == "DenseNet121":
        model = DenseNet_for_Alzheimer(num_classes=args.num_classes)
    elif args.model_type == "EfficientNet":
        model = EfficientNet_for_Alzheimer(num_classes=args.num_classes)
    elif args.model_type == "VGG16":
        model = VGG_for_Alzheimer(num_classes=args.num_classes)
    elif args.model_type == "MobileNetV2":
        model = MobileNet_for_Alzheimer(num_classes=args.num_classes)
    elif args.model_type == "ResNet50":
        model = ResNet50_for_Alzheimer(num_classes=args.num_classes)
    elif args.model_type == "TinyViT_BEiT":
        model = TinyViT_BEiT(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif args.model_type == "TinyViT_DeiT_with_Inception":
        model = TinyViT_DeiT_with_Inception(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif args.model_type == "TinyViT_DeiT_with_ModifiedInception":
        model = TinyViT_DeiT_with_ModifiedInception(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif args.model_type == "TinyViT_with_Inception_Advanced":
        model = TinyViT_with_Inception_Advanced(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone,
            gradient_checkpointing=True
        )
    elif args.model_type == "TinyViT_with_ModifiedInception_Advanced":
        model = TinyViT_with_ModifiedInception_Advanced(
            img_size=args.img_size,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            freeze_backbone=not args.unfreeze_backbone,
            gradient_checkpointing=True
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    model.to(args.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: \t{num_params:,}")
    
    # Enable torch.backends optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if args.fp16:
        # Keep the model in float32, we'll use AMP instead of full FP16
        model = model.float()
        # Ensure all batch norm and layer norm stay in float32
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                module.float()
    
    return args, model, optimizer
    
    return args, model, optimizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_labels = [], []
    epoch_iterator = tqdm(
        test_loader,
        desc = "Validating.. (loss=X.X)",
        bar_format = "{l_bar}{r_bar}",
        dynamic_ncols = True
    )
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, (x, y) in enumerate(epoch_iterator):
        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad(): 
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_labels[0] = np.append(
                all_labels[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validation... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_labels = all_preds[0], all_labels[0]
    accuracy = simple_accuracy(all_preds, all_labels)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    writer.add_scalar("test/loss", scalar_value=eval_losses.avg, global_step=global_step)
    return accuracy, eval_losses.avg

def get_optimizer(args, model):
    if args.model_type in ['TinyViT_DeiT_with_Inception', 'TinyViT_DeiT_with_ModifiedInception']:
        return get_optimizer_for_deit(args, model)
    # elif args.model_type in ['TinyViT_with_Inception_Advanced', 'TinyViT_with_ModifiedInception_Advanced']:
    #     return get_optimizer_for_advanced(args, model)
    elif args.model_type in ['VanillaViT', 'VanillaViT_with_Inception', 'VanillaViT_with_ModifiedInception']:
        optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999))
    elif args.model_type in ["TinyViT_DeiT", "TinyViT_DeiT_with_Inception", "TinyViT_DeiT_with_ModifiedInception"]:
        max_lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            steps_per_epoch=len(train_loader),
            epochs=args.num_epochs,
            pct_start=0.2,  
            div_factor=10,
            final_div_factor=1e3,
            anneal_strategy='cos'
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    return optimizer

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        label_a, label_b = labels, labels[index]
        return mixed_batch, label_a, label_b, lam

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model)
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(args, model, optimizer):
    """Training"""
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name = args.model_type
    eval_dir = os.path.join(args.output_dir, "eval", model_name)
    os.makedirs(eval_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=eval_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    elif args.dataset == "alzheimers":
        train_loader, val_loader, _ = load_alzheimers_data(
            args.data_dir,
            batch_size=args.train_batch_size,
            dataset_type=args.dataset_type,
            val_split=args.val_split
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(args, optimizer, train_loader)
    
    # Initialize metrics
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    best_val_acc = 0.0
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(args.output_dir, args.model_type)
    
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    # Initialize AMP gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # Initialize Mixup and EMA
    mixup = Mixup(alpha=0.2)
    model_ema = ModelEMA(model)
    
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    for epoch in range(args.num_epochs):
        train_losses.reset()
        train_acc.reset()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for step, (images, labels) in enumerate(pbar):
            # Keep inputs in float32, AMP will handle the conversion
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            
            # Clear memory cache periodically
            if step % 10 == 0:
                torch.cuda.empty_cache()
            
            # Apply Mixup if enabled
            if args.use_mixup:
                images, labels_a, labels_b, lam = mixup(images, labels)
            
            # Use automatic mixed precision context
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(images)
                if args.use_mixup:
                    loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
                else:
                    loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Unscale gradients and clip norm
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Step optimizer and scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                # Update EMA model if enabled
                if args.use_ema and model_ema is not None:
                    model_ema.update(model)
                
                # Update metrics
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = (predicted == labels).float().mean()
                    train_losses.update(loss.item())
                    train_acc.update(accuracy.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{train_losses.avg:.4f}',
                    'acc': f'{train_acc.avg:.4f}'
                })
                
                # Log metrics
                if step % args.logging_steps == 0:
                    writer.add_scalar("train/loss", scalar_value=train_losses.avg, global_step=step)
                    writer.add_scalar("train/accuracy", scalar_value=train_acc.avg, global_step=step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)

        # Validation phase
        val_loss, val_acc = validate(args, model, val_loader, criterion)
        
        early_stopping(val_loss)
        if early_stopping.stop:
            logger.info("Early stopping triggered")
            break
        # Save periodic checkpoint
        if (epoch + 1) % args.save_steps == 0:
            save_model(args, model, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),  # Also save scheduler state
                'args': args,  # Save configuration
            }, args.output_dir, is_best=True)  # Add is_best flag
        
        # Log metrics
        metrics_logger.update(
            epoch=epoch,
            train_loss=train_losses.avg,
            train_acc=train_acc.avg,
            val_loss=val_loss,
            val_acc=val_acc
        )
        
        # Save metrics after each epoch
        metrics_logger.save()
        
        writer.add_scalar("val/loss", scalar_value=val_loss, global_step=epoch)
        writer.add_scalar("val/accuracy", scalar_value=val_acc, global_step=epoch)
        
        # Log to console
        logger.info(f'Epoch {epoch+1}: '
                   f'Train Loss={train_losses.avg:.4f}, '
                   f'Train Acc={train_acc.avg:.4f}, '
                   f'Val Loss={val_loss:.4f}, '
                   f'Val Acc={val_acc:.4f}')
        
        if early_stopping(val_loss):
            logger.info("Early stopping")
            break

    writer.close()
    logger.info(f"Best Accuracy for {model_name}: \t%f" % best_val_acc)
    logger.info(f"End Training for {model_name}")

    return best_val_acc

def validate(args, model, val_loader, criterion):
    model.eval()
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
        for images, labels in val_loader:
            # Keep inputs in float32, AMP will handle the conversion
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean()
            
            val_losses.update(loss.item())
            val_acc.update(accuracy.item())
    
    model.train()
    return val_losses.avg, val_acc.avg

def train_epoch(model, train_loader, optimizer, criterion, scheduler, device, args):
    mixup = Mixup(alpha=0.2)
    model.train()
    total_loss = 0
    steps_per_update = 4  # Accumulate over 4 steps
    optimizer.zero_grad()
    
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) / steps_per_update
        loss.backward()
        
        if (idx + 1) % steps_per_update == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * steps_per_update

def get_optimizer_for_deit(args, model):
    # Increase base learning rate and use layer-wise learning rates
    backbone_params = {'params': [], 'lr': args.learning_rate * 0.1}  # Lower LR for pretrained
    head_params = {'params': [], 'lr': args.learning_rate * 5.0}      # Higher LR for new parts
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params['params'].append(param)
        else:
            head_params['params'].append(param)
    
    optimizer = torch.optim.AdamW(
        [backbone_params, head_params],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer

def get_scheduler(args, optimizer, train_loader):
    num_training_steps = args.num_epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    if args.model_type in ["TinyViT_DeiT", "TinyViT_DeiT_with_Inception", "TinyViT_DeiT_with_ModifiedInception"]:
        # For DeiT models, use OneCycleLR
        max_lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            steps_per_epoch=len(train_loader),
            epochs=args.num_epochs,
            pct_start=0.2,  # Faster warmup
            div_factor=10,  # Less aggressive initial lr reduction
            final_div_factor=1e3,
            anneal_strategy='cos'
        )
    else:
        # For other models, use WarmupCosineScheduler
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=num_warmup_steps,
            t_total=num_training_steps
        )
    
    return scheduler

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * float(step) / float(max(1, self.warmup_steps)) 
                   for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            return [base_lr * 0.5 * (1. + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]

def main():
    parser = argparse.ArgumentParser()
    # Add these arguments before the existing ones
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--unfreeze_backbone", action="store_true",
                       help="Whether to unfreeze the pretrained backbone")
    parser.add_argument("--num_classes", type=int, default=4,
                       help="Number of classes in the dataset")
    parser.add_argument("--img_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--num_channels", type=int, default=3,
                       help="Number of input channels")
    parser.add_argument("--patch_size", type=int, default=16,
                       help="Size of image patches")
    parser.add_argument("--embeddingdim", type=int, default=768,
                       help="Embedding dimension")
    parser.add_argument("--mlp_size", type=int, default=3072,
                       help="MLP hidden dimension")
    parser.add_argument("--num_transformer_layer", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fraction of training data to use for validation")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Ratio of total training steps to use for warmup")
    
    # Existing arguments
    parser.add_argument("--name", required=True,
                       help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "alzheimers"],
                       default="alzheimers", help="Which downstream task.")
    parser.add_argument("--model_type", 
                    choices=[
                        "VanillaViT", "VanillaViT_with_Inception",
                        "VanillaViT_with_ModifiedInception", "ResNet50",
                        "DenseNet121", "EfficientNet", "VGG16", "MobileNetV2",
                        "TinyViT", "TinyViT_with_Inception", "TinyViT_with_ModifiedInception",
                        "TinyViT_DeiT", "TinyViT_Swin", "TinyViT_ConvNeXt", "TinyViT_BEiT",
                        "TinyViT_DeiT_with_Inception", "TinyViT_DeiT_with_ModifiedInception",
                        "TinyViT_with_Inception_Advanced",
                        "TinyViT_with_ModifiedInception_Advanced"
                    ],
                    default="VanillaViT",
                    help="Which model architecture to use")
    parser.add_argument("--output_dir", default="output", type=str,
                       help="The output directory where checkpoints will be written.")
    parser.add_argument("--dataset_type", choices=["Original", "Augmented"],
                       default="Original", help="Which dataset type to use")

    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int, help="Run prediction on validation set every so many steps.")

    parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--data_dir", default="./data/alzheimers", type=str, help="Path to the dataset directory")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit mixed precision.")

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Add optimization arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true",
                       help="Pin memory for faster data transfer")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Number of batches to prefetch")
    
    # Add new arguments
    parser.add_argument("--use_mixup", action="store_true",
                       help="Whether to use mixup augmentation")
    parser.add_argument("--use_ema", action="store_true",
                       help="Whether to use model EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                       help="Decay rate for EMA")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                       help="Alpha parameter for mixup")
    
    args = parser.parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_proccess_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    set_seed(args)
    args, model, optimizer = setup(args)
    train(args, model, optimizer)

if __name__ == "__main__":
    main()
