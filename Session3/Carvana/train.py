"""
Main file for training UNET model on Carvana dataset
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNET
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 2
WEIGHT_DECAY = 0
EPOCHS = 2
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 720
LOAD_MODEL_FILE = "Session3/Carvana/model_2.pth.tar"

# Define paths
train_path = 'C:/Datasets/Carvana/data/train_small'
train_masks_path = 'C:/Datasets/Carvana/data/train_masks_small'
val_path = 'C:/Datasets/Carvana/data/valid_small'
val_mask_path = 'C:/Datasets/Carvana/data/valid_masks_small'
saved_val_results = 'C:/Datasets/Carvana/data/saved_images/'
os.makedirs(saved_val_results, exist_ok=True)

# Augmentations: normalize, resize, horizontal and vertical flip
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value = 255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0], max_pixel_value = 255.0),
            ToTensorV2(),
])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    for step, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # Add channel dimension to target if it's missing
        if y.dim() == 3:  # If y has shape [batch_size, height, width]
            y = y.unsqueeze(1)  # Add a channel dimension, resulting in shape [batch_size, 1, height, width]
        # Forward Pass
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())


def main():
    # Get training and validation data loaders
    train_loader, val_loader = get_loaders(
        train_dir = train_path,
        train_maskdir = train_masks_path,
        val_dir = val_path,
        val_maskdir = val_mask_path,
        batch_size = BATCH_SIZE,
        train_transform = train_transform,
        val_transform = val_transform,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY
    )
    
    # Setup model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    # Setup loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model)

    # Create training loop
    for epoch in range(EPOCHS):  # gives batch data
        print(f"Epoch {epoch+1} start")
        train_fn(train_loader, model, optimizer, loss_fn)
        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}
        save_checkpoint(checkpoint, filename='LOAD_MODEL_FILE')

        # Check validation accuracy
        val_acc = check_accuracy(val_loader, model, device=DEVICE)
        print(f"Validation Accuracy: {val_acc}")

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()