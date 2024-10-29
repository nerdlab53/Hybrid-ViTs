from torch.utils.data import DataLoader
from dataset_utils.alzheimers_dataset import AlzheimersDataset
from torchvision import transforms

def load_alzheimers_data(data_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AlzheimersDataset(
        root_dir=f"{data_dir}/train",
        transform=transform
    )
    
    test_dataset = AlzheimersDataset(
        root_dir=f"{data_dir}/test",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader