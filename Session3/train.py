"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
# tqdm means "progress" in Arabic (taqadum): make your loops show a smart progress meter
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 4
WEIGHT_DECAY = 0
EPOCHS = 5
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# Create train function
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True) # Wrap the train_loader with tqdm
    mean_loss = []
    for step, (x, y) in enumerate(loop):
        x = x.to(DEVICE)    
        y = y.to(DEVICE)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update progress bar
        loop.set_postfix(loss = loss.item())
    
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
    # Setup model
    model = Yolov1(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    # Setup loss function
    loss_fn = YoloLoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    print("Starting training process...")

    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # Create training loop
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch+1, "\n")
        train_pred_boxes, train_true_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        train_MAP = mean_average_precision(train_pred_boxes, train_true_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {train_MAP}")
        
        train_fn(train_loader, model, optimizer, loss_fn)
    
    # Save checkpoint when done 
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

    print("Training process completed.")

if __name__ == "__main__":
    main()