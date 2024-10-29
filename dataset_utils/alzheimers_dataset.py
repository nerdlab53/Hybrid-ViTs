import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
import os
from typing import List, Tuple, Optional
from PIL import Image
from pathlib import Path


class AlzheimersDataset(Dataset):

    def __init__(self, root_dir: str, dataset_type: str = "Original", transform: Optional[transforms.Compose] = None, 
                 image_size: Tuple[int, int] = (224, 224), split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.dataset_type = dataset_type
        self.split = split
        
        # Adjust the path based on split
        data_path = os.path.join(root_dir, dataset_type, 'train' if split == 'train' else 'test')
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        
        # Default Transform if None is specified
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, int]:
        
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image : {img_path} : {e}")
            image = Image.new('RGB', self.image_size, color='black')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]
    
    def get_class_names(self) -> List[str]:
        return sorted(os.listdir(self.root_dir))
    
    def get_num_classes(self) -> int:
        return len(sorted(os.listdir(self.root_dir)))
    
    def get_sample_distribution(self) -> dict:
        dist = {cls:0 for cls in sorted(os.listdir(self.root_dir))}
        for _, label in self.labels:
            dist[sorted(os.listdir(self.root_dir))[label]] += 1
        return dist
