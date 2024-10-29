from torch.utils.data import DataLoader
from dataset_utils.alzheimers_dataset import AlzheimersDataset
from torchvision import transforms
import torch

def load_alzheimers_data(data_dir, batch_size=32, num_workers=4, dataset_type="Original", val_split=0.1):
    """Load Alzheimer's dataset with train/val splits"""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = AlzheimersDataset(
        root_dir=data_dir,
        dataset_type=dataset_type,
        transform=transform_train
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    # Create train/val splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create custom Subset class that can modify transforms
    class TransformSubset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, transform=None):
            super().__init__(dataset, indices)
            self.transform = transform
        
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            if self.transform:
                x = self.transform(x)
            return x, y
    
    # Apply different transforms to validation set
    val_dataset = TransformSubset(full_dataset, val_dataset.indices, transform_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, None  # Return None for test_loader as we don't have a separate test set