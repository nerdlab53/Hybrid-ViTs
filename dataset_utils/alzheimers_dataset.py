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

    def __init__(self, root_dir: str, dataset_type: str = "Original", transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir (str): Directory with all the images
            dataset_type (str): "Original" or "Augmented" to specify which dataset to use
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Map dataset_type to actual folder names
        dataset_map = {
            "Original": "OriginalDataset",
            "Augmented": "AugmentedAlzheimerDataset"
        }
        
        # Get the correct folder name
        folder_name = dataset_map.get(dataset_type)
        if not folder_name:
            raise ValueError(f"dataset_type must be one of {list(dataset_map.keys())}")
        
        data_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset path does not exist: {data_path}")
            
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        print(f"Loading data from: {data_path}")
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path) and os.path.isdir(class_path):
                print(f"Processing class {class_name}")
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"Found {len(self.image_paths)} images in {len(set(self.labels))} classes")
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, int]:
        
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image : {img_path} : {e}")
            image = Image.new('RGB', (224, 224), color='black')

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
