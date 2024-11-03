import os
import torch
import csv
import argparse
from torchvision import datasets, transforms
from models.VanillaViT_with_Inception import VanillaViT_with_Inception
from models.VanillaViT_with_ModifiedInception import VanillaViT_with_ModifiedInceptionModule
from models.VanillaViT import VanillaViT
from models.resnet import ResNet50_for_Alzheimer
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer
from utils.data_loader import load_alzheimers_data
from models.TinyViT import TinyViT
from models.TinyViT_with_Inception import TinyViT_with_Inception
from models.TinyViT_with_ModifiedInception import TinyViT_with_ModifiedInception

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, data_loader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_correct += (top5_preds == labels.view(-1, 1)).any(dim=1).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy, all_preds, all_labels

def save_results(results, filename="model_evaluation_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Top-1 Accuracy", "Top-5 Accuracy"])
        for result in results:
            writer.writerow(result)

def get_data_loaders(args):
    if args.dataset == "alzheimers":
        return load_alzheimers_data(args.data_dir, args.batch_size)
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    if args.dataset == "cifar10":
        testset = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        testset = datasets.CIFAR100(root="./data", train=False,
                                  download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return None, test_loader

def load_model(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "alzheimers"], 
                       default="alzheimers")
    parser.add_argument("--data_dir", default="./data/alzheimers", 
                       help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, 
                       default="model_evaluation_results.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="./output",
                       help="Directory containing model checkpoints")
    parser.add_argument("--dataset_type", choices=["Original", "Augmented"],
                       default="Original", help="Which dataset type to use")
    args = parser.parse_args()
    
    _, test_loader = get_data_loaders(args)
    
    models = {
        "VanillaViT": VanillaViT(),
        "VanillaViT_with_Inception": VanillaViT_with_Inception(),
        "VanillaViT_with_ModifiedInception": VanillaViT_with_ModifiedInceptionModule(),
        "ResNet50": ResNet50_for_Alzheimer(),
        "DenseNet121": DenseNet_for_Alzheimer(),
        "EfficientNet-B0": EfficientNet_for_Alzheimer(),
        "VGG16_BN": VGG_for_Alzheimer(),
        "MobileNetV2": MobileNet_for_Alzheimer(),
        "TinyViT": TinyViT(),
        "TinyViT_with_Inception": TinyViT_with_Inception(),
        "TinyViT_with_ModifiedInception": TinyViT_with_ModifiedInception()
    }
    
    results = []
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{name}_final_checkpoint.pth")
        
        if os.path.exists(checkpoint_path):
            model = load_model(model, checkpoint_path, device)
            if model is None:
                print(f"Skipping {name} due to checkpoint loading error")
                continue
        else:
            print(f"No checkpoint found for {name}, using initial weights")
        
        model = model.to(device)
        top1_acc, top5_acc, preds, labels = evaluate_model(model, test_loader, device)
        results.append([name, f"{top1_acc:.2f}%", f"{top5_acc:.2f}%"])
        print(f"{name} - Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%")
    
    save_results(results, args.output_file)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()

