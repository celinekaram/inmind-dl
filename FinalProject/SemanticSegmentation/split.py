import os
import tqdm
import shutil
import torch
import numpy as np

# Define paths to the data directories
images_path = 'organized_data/rgb_images'  # Directory containing RGB images
masks_path = 'organized_data/semantic_segmentation/semantic_image'  # Directory containing mask images
train_path = 'SemanticSegmentation/data/train/images'
train_masks_path = 'SemanticSegmentation/data/train/masks'
val_path = 'SemanticSegmentation/data/val/images'
val_masks_path = 'SemanticSegmentation/data/val/masks'
test_path = 'SemanticSegmentation/data/test/images'
test_masks_path = 'SemanticSegmentation/data/test/masks'

# Create directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(train_masks_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(val_masks_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(test_masks_path, exist_ok=True)

# Set the seed for reproducibility
torch.manual_seed(42)

# List all image files
image_files = sorted(os.listdir(images_path))
total_images = len(image_files)
indices = torch.randperm(total_images).tolist()

# Calculate split indices
train_split = int(0.8 * total_images)
val_split = int(0.9 * total_images)

# Split indices
train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:]

def copy_files(indices, output_dirs):
    """Copy image and mask files to the specified output directories."""
    for idx in tqdm.tqdm(indices, total=len(indices)):
        # File names (assuming images and masks are .png)
        img_name = f"{idx:04d}.png"  # e.g., 0001.png
        mask_name = f"{idx:04d}.png"  # e.g., 0001.png

        # Full paths
        img_path = os.path.join(images_path, img_name)
        mask_path = os.path.join(masks_path, mask_name)

        # Copy files
        shutil.copy(img_path, output_dirs['images'])
        shutil.copy(mask_path, output_dirs['masks'])

# Define output directories for training, validation, and test sets
output_dirs_train = {'images': train_path, 'masks': train_masks_path}
output_dirs_val = {'images': val_path, 'masks': val_masks_path}
output_dirs_test = {'images': test_path, 'masks': test_masks_path}

# Copy files to respective directories
copy_files(train_indices, output_dirs_train)
copy_files(val_indices, output_dirs_val)
copy_files(test_indices, output_dirs_test)