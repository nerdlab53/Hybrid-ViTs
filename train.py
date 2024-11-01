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
from models.VanillaViT import VanillaViT
from models.VanillaViT_with_Inception import VanillaViT_with_Inception
from models.VanillaViT_with_ModifiedInception import VanillaViT_with_ModifiedInceptionModule
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
            num_transformer_layer=args.num_transformer_layer,
            num_classes=args.num_classes
        )
    elif args.model_type == "VanillaViT_with_Inception":
        model = VanillaViT_with_Inception(num_classes=args.num_classes)
    elif args.model_type == "VanillaViT_with_ModifiedInception":
        model = VanillaViT_with_ModifiedInceptionModule(num_classes=args.num_classes)
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
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: \t{num_params:,}")
    return args, model

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
    if args.model_type in ['VanillaViT', 'VanillaViT_with_Inception', 'VanillaViT_with_ModifiedInception']:
        optimizer = torch.optim.SGD(model.parameters(),
                                  lr=args.learning_rate,
                                  momentum=0.9,
                                  weight_decay=args.weight_decay)
    else:  # CNN models
        optimizer = torch.optim.Adam(model.parameters(),
                                   lr=args.learning_rate * 0.1,
                                   weight_decay=args.weight_decay)
    return optimizer

def train(args, model):
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
    optimizer = get_optimizer(args, model)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        t_total=args.num_steps
    )
    
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

    model.zero_grad()
    set_seed(args)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        train_losses.reset()
        train_acc.reset()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for step, (images, labels) in enumerate(pbar):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            
            # Clear memory cache periodically
            if step % 10 == 0:
                torch.cuda.empty_cache()
            
            # Forward pass with mixed precision
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # Update metrics
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).float().mean()
                train_losses.update(loss.item())
                train_acc.update(accuracy.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{train_losses.avg:.4f}',
                    'acc': f'{train_acc.avg:.4f}'
                })
                
                writer.add_scalar("train/loss", scalar_value=train_losses.avg, global_step=step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=step)
                writer.add_scalar("train/accuracy", scalar_value=train_acc.avg, global_step=step)
        
        # Validation phase
        val_loss, val_acc = validate(args, model, val_loader, criterion)
        
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
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean()
            
            val_losses.update(loss.item())
            val_acc.update(accuracy.item())
    
    model.train()
    return val_losses.avg, val_acc.avg

def main():
    parser = argparse.ArgumentParser()
    # Add these arguments before the existing ones
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
    
    # Existing arguments
    parser.add_argument("--name", required=True,
                       help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "alzheimers"],
                       default="alzheimers", help="Which downstream task.")
    parser.add_argument("--model_type", 
                       choices=["VanillaViT", "VanillaViT_with_Inception", 
                               "VanillaViT_with_ModifiedInception", "ResNet50",
                               "DenseNet121", "EfficientNet", "VGG16", "MobileNetV2"],
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
    args, model = setup(args)
    train(args, model)

if __name__ == "__main__":
    main()

