"""
Main file for training ** model on BMW Semantic dataset
"""
import os
import matplotlib as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import CustomModel
from torch.utils.tensorboard import SummaryWriter
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
from utils import calculate_metrics

seed = 123
torch.manual_seed(seed)

# Hyperparameters
num_classes = 10
batch_size = 4
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

def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    # Metric variables
    running_loss = 0.0
    iou_total = 0
    dice_total = 0
    precision_total = 0
    recall_total = 0
    counter = 0
    loop = tqdm(train_loader)
    for step, (image, mask) in enumerate(loop):
        image = image.to(device)
        mask = mask.float().unsqueeze(1).to(device)
        
        # forward pass
        with torch.cuda.amp.autocast():
            prediction = model(image)
            loss = loss_fn(prediction, mask)
            running_loss += loss.item()

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        # Calculate metrics
        iou, dice, precision, recall = calculate_metrics(prediction, mask)
        iou_total += iou
        dice_total += dice
        precision_total += precision
        recall_total += recall
    return running_loss, iou_total, dice_total, precision_total, recall_total
        
        
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
    model = CustomModel(3, num_classes).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # Setup loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # accuracy, iou, training time, inference speed, hyperparams
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
    
    if load_memory:
        load_checkpoint(torch.load(load_model_file), model)

    check_accuracy(val_loader, model, device)
    
    log_dir = "./logs" # directory where TensorBoard logs will be stored
    writer = SummaryWriter(log_dir=log_dir) # log data for visualization in TensorBoard
    print("Training started")
    # Create training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} start")
        running_loss, iou_total, dice_total, precision_total, recall_total = train_fn(train_loader, model, optimizer, loss_fn)
        
        # Calculate average metrics in each epoch
        avg_loss = running_loss / len(train_loader)
        avg_iou = iou_total / len(train_loader)
        avg_dice = dice_total / len(train_loader)
        avg_precision = precision_total / len(train_loader)
        avg_recall = recall_total / len(train_loader)

        # Log metrics to TensorBoard associated with the current epoch number
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Metrics/IoU", avg_iou, epoch)
        writer.add_scalar("Metrics/Dice", avg_dice, epoch)
        writer.add_scalar("Metrics/Precision", avg_precision, epoch)
        writer.add_scalar("Metrics/Recall", avg_recall, epoch)
        
        writer.close()
        
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