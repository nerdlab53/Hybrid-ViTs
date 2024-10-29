import torch
import csv
from torchvision.transforms import functional as TF
from models.VanillaViT_with_Inception import VanillaViT_with_Inception
from models.VanillaViT_with_ModifiedInception import VanillaViT_with_ModifiedInceptionModule
from models.VanillaViT import VanillaViT
from models.resnet import ResNet50_for_Alzheimer
from models.densenet import DenseNet_for_Alzheimer
from models.efficientnet import EfficientNet_for_Alzheimer
from models.vgg import VGG_for_Alzheimer
from models.mobilenet import MobileNet_for_Alzheimer
from utils.data_loader import load_data, load_alzheimers_data
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def save_results(results, filename="model_evaluation_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Top-1 Accuracy", "Top-5 Accuracy"])
        for result in results:
            writer.writerow(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "alzheimers"], default="alzheimers")
    parser.add_argument("--data_dir", default="./data/alzheimers", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    if args.dataset == "alzheimers":
        train_loader, test_loader = load_alzheimers_data(args.data_dir, args.batch_size)
    else:
        
        pass
    
    models = {
        "VanillaViT_with_Inception": VanillaViT_with_Inception(),
        "VanillaViT_with_ModifiedInception": VanillaViT_with_ModifiedInceptionModule(),
        "VanillaViT": VanillaViT(),
        "ResNet50": ResNet50_for_Alzheimer(),
        "DenseNet121": DenseNet_for_Alzheimer(),
        "EfficientNet-B0": EfficientNet_for_Alzheimer(),
        "VGG16_BN": VGG_for_Alzheimer(),
        "MobileNetV2": MobileNet_for_Alzheimer()
    }
    
    results = []
    for name, model in models.items():
        model = model.to(device)
        accuracy, preds, labels = evaluate_model(model, test_loader, device)
        results.append([name, accuracy])
        print(f"{name} - Accuracy: {accuracy:.2f}%")
    
    save_results(results)

if __name__ == "__main__":
    main()

