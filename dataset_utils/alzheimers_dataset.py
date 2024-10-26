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

    def __init__(
        self,
        root_dir : str,
        transform : Optional[transforms.compose] : None,
        image_size : Tuple[int, int] : (224, 224)
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size

        self.classes = sorted([d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)])

        self.class_to_idx = {cls_name : i for i, cls_name in enumerate(self.classes)}
        
        self.images: List[Tuple[str, int] = []
        self._load_dataset()
        # Default Transform if None is specified
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
    
    def _load_dataset(self):

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = self.root_dir / class_name

            for img_path in class_dir.glob(*):
                if img_path.suffix.lower() in valid_extensions:
                    self.images.append((str(img_path), class_idx))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, int]:
        
        img_path, label = self.images[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image : {img_path} : {e}")
            image = Image.new('RGB', self.image_size, color='black')

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class_names(self) -> List[str]:
        return self.classes
    
    def get_num_classes(self) -> int:
        return len(self.classes)
    
    def get_sample_distribution(self) -> dict:
        dist = {cls:0 for cls in self.classes}
        for _, label in self.images:
            dist[self.classes[label]] += 1
        return dist
