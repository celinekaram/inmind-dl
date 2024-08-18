import os
import tqdm
import shutil
import torch

# Paths to directories
images_dir = 'organized_data/rgb_images'
labels_dir = 'organized_data/bounding_boxes/bb_yolo_format'

# Base directory for outputs
base_dir = 'ObjectDetection/data/'

# Output directories
output_dirs = {
    'train': {
        'images': os.path.join(base_dir, 'train/images'),
        'labels': os.path.join(base_dir, 'train/labels')
    },
    'val': {
        'images': os.path.join(base_dir, 'val/images'),
        'labels': os.path.join(base_dir, 'val/labels')
    },
    'test': {
        'images': os.path.join(base_dir, 'test/images'),
        'labels': os.path.join(base_dir, 'test/labels')
    }
}

# Set the seed for reproducibility
torch.manual_seed(42)

# Create directories if they don't exist
def create_directories():
    for key, dirs in output_dirs.items():
        os.makedirs(dirs['images'], exist_ok=True)
        os.makedirs(dirs['labels'], exist_ok=True)

def copy_files(indices, output_dir):
    """Copy image and label files to the specified output directory."""
    for idx in tqdm.tqdm(indices, total=len(indices)):
        # File names (assuming images are .png and labels are .txt)
        img_name = f"{idx:04d}.png"  # e.g., 0001.png
        lbl_name = f"{idx:04d}.txt"  # e.g., 0001.txt

        # Full paths
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, lbl_name)

        # Copy files
        shutil.copy(img_path, output_dir['images'])
        shutil.copy(lbl_path, output_dir['labels'])