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
from utils.data_loader import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, data_loader):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, top5 = outputs.topk(5, 1, True, True)
            top1_correct += (top5[:, 0:1] == labels.view(-1, 1)).sum().item()
            top5_correct += (top5 == labels.view(-1, 1)).any(dim=1).sum().item()
            total += labels.size(0)

    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100
    return top1_accuracy, top5_accuracy

def save_results(results, filename="model_evaluation_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Top-1 Accuracy", "Top-5 Accuracy"])
        for result in results:
            writer.writerow(result)

def main():
    data_loader = load_data()
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
        model = model.to(device)  # Add device handling
        top1_accuracy, top5_accuracy = evaluate_model(model, data_loader)
        results.append([name, top1_accuracy, top5_accuracy])
        print(f"{name} - Top-1 Accuracy: {top1_accuracy:.2f}%, Top-5 Accuracy: {top5_accuracy:.2f}%")

    save_results(results)

if __name__ == "__main__":
    main()
