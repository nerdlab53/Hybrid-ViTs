import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import timedelta
from models.VanillaViT import VanillaViT
from models.VanillaViT_with_Inception import VanillaViT_with_Inception
from models.VanillaViT_with_ModifiedInception import VanillaViT_with_ModifiedInception
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer
from utils.scheduler import WarmupCosineScheduler
from dataset_utils.alzheimers_dataset import AlzheimersDataset

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
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint_{global_step}.bin")
    torch.save(model_to_save.save_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    if args.model_type == 'VanillaViT':
        model = VanillaViT(num_classes=args.num_classes)
    elif args.model_type == 'VanillaViT_with_Inception':
        model = VanillaViT_with_Inception(num_classes=args.num_classes)
    elif args.model_type == 'VanillaViT_with_ModifiedInception':
        model = VanillaViT_with_ModifiedInception(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unkown model type : {args.model_type}")
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters : \t%2.1fM" % (num_params / 1000000))
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

def train(args, model):
    """Training"""
    
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
        trainset = AlzheimersDataset(root_dir=args.data_dir + "/train", transform=transform_train)
        testset = AlzheimersDataset(root_dir=args.data_dir + "/test", transform=transform_test)

    
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()

    global_step, best_acc = 0, 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    
    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training {model_name} (X / X Steps) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True
        )

        for step, (x, y) in enumerate(epoch_iterator):
            x, y = x.to(args.device), y.to(args.device)
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    f"Training {model_name} ({global_step} / {t_total} steps) (loss={losses.val:.5f})"
                )

                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                train_losses.append((global_step, losses.val))

                if global_step % args.eval_every == 0:
                    accuracy, val_loss = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, global_step)
                        best_acc = accuracy
                    model.train()
                    
                    train_accuracies.append((global_step, accuracy))
                    val_losses.append((global_step, val_loss))
                    val_accuracies.append((global_step, accuracy))

                if global_step % t_total == 0:
                    break
        
        losses.reset()
        
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info(f"Best Accuracy for {model_name}: \t%f" % best_acc)
    
    np.save(os.path.join(eval_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(eval_dir, "train_accuracies.npy"), np.array(train_accuracies))
    np.save(os.path.join(eval_dir, "val_accuracies.npy"), np.array(val_accuracies))
    np.save(os.path.join(eval_dir, "val_losses.npy"), np.array(val_losses))

    logger.info(f"End Training for {model_name}")

    return best_acc

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "alzheimers"], default="cifar10", help="Which downstream task.")
    parser.add_argument("--model_type", choices=["VanillaViT", "VanillaViT_with_Inception", "VanillaViT_with_ModifiedInception", "DenseNet121", "EfficientNet", "VGG16", "MobileNetV2"],
                        default="VanillaViT", help="Which model to use.")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int, help="Run prediction on validation set every so many steps.")

    parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--data_dir", default="./data/alzheimers", type=str, help="Path to the dataset directory")
    
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

