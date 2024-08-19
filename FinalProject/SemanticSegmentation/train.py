"""
Main file for training ** model on BMW Semantic dataset
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from UNETmodel import CustomModel
# from model import CustomModel
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

seed = 123
torch.manual_seed(seed)

# Hyperparameters
num_classes = 10
batch_size = 16
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
num_workers = 2
pin_memory = True

IMAGE_HEIGHT = 1280 # originally 1280
IMAGE_WIDTH = 720 # originally 720
load_model_file = "model_1.pth.tar"
load_memory = False

# Define paths
train_path = 'SemanticSegmentation/data/train/images'
train_masks_path = 'SemanticSegmentation/data/train/masks'
val_path = 'SemanticSegmentation/data/val/images'
val_masks_path = 'SemanticSegmentation/data/val/masks'
test_path = 'SemanticSegmentation/data/test/images'
test_masks_path = 'SemanticSegmentation/data/test/masks'
saved_val_results = 'SemanticSegmentation/val/results'
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
    for step, (image, mask) in enumerate(loop):
        image = image.to(device)
        mask = mask.to(device)
        # Forward Pass
        y_pred = model(image)
        loss = loss_fn(y_pred, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())


def main():
    # Get training and validation data loaders
    train_loader, val_loader = get_loaders(
        train_path,
        train_masks_path,
        val_path,
        val_masks_path,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory
    )
    # Setup model
    model = CustomModel(in_channels=3, out_channels=num_classes).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # Setup loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # accuracy, iou, training time, inference speed, hyperparams
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
    
    if load_memory:
        load_checkpoint(torch.load(load_model_file), model)

    # Create training loop
    for epoch in range(num_epochs):  # gives batch data
        print(f"Epoch {epoch+1} start")
        train_fn(train_loader, model, optimizer, loss_fn)
        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}
        save_checkpoint(checkpoint, filename='NEW_MODEL_FILE')

        # Check validation accuracy
        val_acc = check_accuracy(val_loader, model, device=device)
        print(f"Validation Accuracy: {val_acc}")

        save_predictions_as_imgs(val_loader, model, folder = saved_val_results, device=device)

if __name__ == "__main__":
    main()