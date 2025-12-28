import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import cv2 

class IDCDataset(Dataset):
    """Dataset for IDC breast cancer patches."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 stain_normalizer=None,
                 target_size: int = 56):
        """
        Args:
            data_dir: Root directory with class folders (0, 1)
            split: 'train', 'val', or 'test'
            transform: Albumentations transform
            stain_normalizer: Stain normalization object
            target_size: Target image size for ViT
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.stain_normalizer = stain_normalizer
        self.target_size = target_size
        
        # Load file paths and labels
        self.samples = []
        for label in [0, 1]:
            class_dir = self.data_dir / str(label)
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), label))
        
        print(f"{split.upper()} set: {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Stain normalization
        if self.stain_normalizer is not None:
            image = self.stain_normalizer.transform(image)
        
        # Resize for ViT if needed
        if image.shape[0] != self.target_size:
            image = cv2.resize(image, (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_LINEAR)
        
        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(label, dtype=torch.float32)
        image = image.contiguous()
        return image, label
