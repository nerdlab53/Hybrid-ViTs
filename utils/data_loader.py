from torch.utils.data import DataLoader
from dataset_utils.alzheimers_dataset import AlzheimersDataset
from torchvision import transforms
import torch

def load_alzheimers_data(data_dir, batch_size=32, num_workers=4, dataset_type="Original", val_split=0.1):
    """Load Alzheimer's dataset with train/val/test splits
    
    Args:
        data_dir: Path to dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        dataset_type: "Original" or "Augmented"
        val_split: Fraction of training data to use for validation
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    train_dataset = AlzheimersDataset(
        root_dir=data_dir,
        dataset_type=dataset_type,
        transform=transform_train,
        split='train'
    )
    
    # Calculate split sizes
    total_size = len(train_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    # Split into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a new validation dataset with test transforms
    val_dataset = AlzheimersDataset(
        root_dir=data_dir,
        dataset_type=dataset_type,
        transform=transform_test,
        split='train'  # Still using train split, but different subset
    )
    
    # Load test dataset
    test_dataset = AlzheimersDataset(
        root_dir=data_dir,
        dataset_type=dataset_type,
        transform=transform_test,
        split='test'
    )
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader