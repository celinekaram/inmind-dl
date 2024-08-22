"""
Main file for training UNET model on BMW Semantic dataset
"""
import warnings

# Suppress specific warning related to Albumentations version checking
warnings.filterwarnings("ignore", message="Error fetching version info")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from UNETmodel import CustomModel
from torch.utils.tensorboard import SummaryWriter
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import calculate_metrics

seed = 123
torch.manual_seed(seed)

# Hyperparameters
num_classes = 10
batch_size = 2
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1
num_workers = 2
pin_memory = True
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# Define paths
train_path = 'SemanticSegmentation/data/train/images'
train_masks_path = 'SemanticSegmentation/data/train/masks'
val_path = 'SemanticSegmentation/data/val/images'
val_masks_path = 'SemanticSegmentation/data/val/masks'
saved_val_results = 'SemanticSegmentation/data/val/results'

models_dir = 'SemanticSegmentation/models'
load_model_file = os.path.join(models_dir, "best.pt")
load_memory = False

# Create directories if they don't exist
os.makedirs(saved_val_results, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")

    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate for training')
    parser.add_argument('--load_memory', type=str, choices=['true', 'false'], default='false', help='Whether to load memory. Use "true" or "false". Default is "false".')
    parser.add_argument('--best_checkpoint', type=str, default=load_model_file, help='Path to save the best checkpoint')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='Number of epochs to train for')
    parser.add_argument('--num_workers', type=int, default=num_workers, help='Number of workers for data loading')
    parser.add_argument('--IMAGE_HEIGHT', type=int, default=IMAGE_HEIGHT, help='Height of input images')
    parser.add_argument('--IMAGE_WIDTH', type=int, default=IMAGE_WIDTH, help='Width of input images')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Access the arguments
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    best_checkpoint = args.best_checkpoint
    num_epochs = args.num_epochs
    num_workers = args.num_workers
    image_height = args.IMAGE_HEIGHT
    image_width = args.IMAGE_WIDTH
    
    # Augmentations: normalize, resize, horizontal and vertical flip
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value = 255.0),
        ToTensorV2(transpose_mask=True), # transpose 3D input mask from [height, width, num_channels] to [num_channels, height, width]
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0], max_pixel_value = 255.0),
        ToTensorV2(transpose_mask=True),
    ])

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
    model = CustomModel(3, num_classes).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # Setup loss function
    loss_fn = nn.CrossEntropyLoss()
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
    
    if load_memory:
        load_checkpoint(torch.load(best_checkpoint), model)
    
    # Tensorboard
    log_dir = "SemanticSegmentation/logs" # directory where TensorBoard logs will be stored
    writer = SummaryWriter(log_dir=log_dir) # log data for visualization in TensorBoard

    print("Training started")
    # Create training loop
    for epoch in range(num_epochs):
        model.train() # training mode
        # Initialize metrics
        running_loss = 0.0
        iou_total = 0.0
        precision_total = 0.0
        recall_total = 0.0
        print(f"Epoch {epoch+1} start")
        loop = tqdm(train_loader)
        for step, (image, mask) in enumerate(loop):
            image = image.to(device) # [batch_size, rgb_channels=3, height, width]
            mask = mask.float().to(device) # [batch_size, num_classes, height, width]
            # forward pass
            prediction = model(image) # [batch_size, num_classes, height, width]
            loss = loss_fn(prediction, mask)
            running_loss += loss.item()
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            
            # Calculate metrics
            iou, precision, recall = calculate_metrics(prediction, mask)
            iou_total += iou
            precision_total += precision
            recall_total += recall
            
            # Calculate average metrics in each epoch
            avg_loss = running_loss / len(train_loader)
            avg_iou = iou_total / len(train_loader)
            avg_precision = precision_total / len(train_loader)
            avg_recall = recall_total / len(train_loader)

            # Log metrics to TensorBoard associated with the current epoch number
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Metrics/IoU", avg_iou, epoch)
            writer.add_scalar("Metrics/Precision", avg_precision, epoch)
            writer.add_scalar("Metrics/Recall", avg_recall, epoch)
            
        writer.close()
            
        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}
        try:
            save_checkpoint(checkpoint, filename=os.path.join(models_dir, f"epoch_{epoch}.pt"))
        except RuntimeError as e:
            print(f"Error saving checkpoint: {e}")
        
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()