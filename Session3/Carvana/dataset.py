"""
Creates a Pytorch dataset to load the Carvana dataset
"""

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform =  transform
        self.images = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_dir = os.path.join(self.image_dir, self.images[idx])
        mask_dir = os.path.join(self.mask_dir, self.images[idx]).replace(".jpg", "_mask.gif")
        image = np.array(Image.open(image_dir).convert("RGB"))
        mask = np.array(Image.open(mask_dir).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0 # white: 1, black: 0
        
        if self.transform:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations ["image"]
            mask = augmentations ["mask"]
        return image, mask