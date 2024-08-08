"""
This script is run once to split the training and validation set from the initial dataset
"""

import os
import shutil
import random

# Define paths
image_folder = 'data/image'
mask_folder = 'data/masks'
train_path = 'data/train'
train_masks_path = 'data/train_masks'
val_path = 'data/valid'
val_mask_path = 'data/valid_masks'

# Create new folders if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(train_masks_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(val_mask_path, exist_ok=True)

# Get list of image files and shuffle them
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
random.shuffle(image_files)

# Calculate split index
split_idx = int(0.8 * len(image_files))

# Split into training and validation sets
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

# Function to copy images and corresponding masks
def copy_files(image_list, dest_image_folder, dest_mask_folder):
    for image_file in image_list:
        image_name = os.path.splitext(image_file)[0]
        mask_file = f"{image_name}_mask.gif"
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(dest_image_folder, image_file))
        shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(dest_mask_folder, mask_file))

# Copy files to respective folders
copy_files(train_images, train_path, train_masks_path)
copy_files(val_images, val_path, val_mask_path)

print(f'Training set: {len(train_images)} images')
print(f'Validation set: {len(val_images)} images')