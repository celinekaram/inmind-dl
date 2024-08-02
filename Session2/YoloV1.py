import numpy as np
import torch
from torch import nn
from torchsummary import summary
import albumentations as A
import cv2
import os

# # Pascal VOC detection dataset (1000 images)
# images_path = "./Session1/images"
# labels_path = "./Session1/labels"
# output_yolo_images_path = "./Session2/augmented_yolo_images"
# output_yolo_labels_path = "./Session2/augmented_yolo_labels"

# # Create output directories if they don't exist
# os.makedirs(output_yolo_images_path, exist_ok=True)
# os.makedirs(output_yolo_labels_path, exist_ok=True)

# grid size S x S = 7 x 7
S = 7
# Bounding boxes predicted in each grid cell
B = 2
# Labelled classes
C = 20

image_path = './Session1/images/000001.jpg'
label_path = "./Session1/labels/000001.txt"

image = cv2.imread(image_path)

    # Open the label file
with open(label_path, 'r') as file:
    lines = file.readlines()
 
# Initialize an empty list to store bounding boxes
bboxes = []

# Iterate through each line
for line in lines:
    # Split the line by whitespace to get class and coordinates
    data = line.strip().split()
    # Extract the coordinates and convert them to floats
    # class x_center y_center width height
    bbox = [float(coord) for coord in data[1:]]
    bbox.append(data[0])
    # Append the bounding box to the list of bounding boxes
    bboxes.append(bbox)
    
# Data augmentation   
# Resize the input image to 448 x 448
resize = A.Resize(448, 448)
# Random scaling and translations of up to 20%
random_scale = A.ShiftScaleRotate(scale_limit=0.2, shift_limit=0.2, rotate_limit=0)
# Randomly adjust exposure and saturation of the image by up to a factor of 15 in the HSV colorspace
hue_saturation_value = A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15)

# Compose all the transformations
transform = A.Compose([
    random_scale,
    hue_saturation_value,
    resize
], bbox_params=A.BboxParams(format='yolo'))

# Apply the transformation
transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
    
# Ground truth bounding box: 
# Each bounding box consists of 5 predictions: x_center, y_center, width, height, and IOU (confidence)

# Draw bounding boxes on the original image
for bbox in bboxes:
    x_center, y_center, width, height, _ = bbox
    img_height, img_width = image.shape[:2]
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

# Define YOLO v1 architecture
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        
        self.features = nn.Sequential(
            # 1st Layer: Conv Layer
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0), # 7x7x64

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 2nd Layer: Conv Layer
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1),  # 3x3x192

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 3rd Layer: Conv Layers
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 4th Layer: Conv Layers
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 5th Layer: Conv Layers
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
            
            # If pretraining, stop here and add avgpooling
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            
            # If not pretraining, continue
            # 6th Layer: Conv Layers
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1),
 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2),
 
            
            # 7th Layer: Conv Layers
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1),
 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1),
 
        )
        
        # 2 Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7*7*1024, out_features=4096),
 
            # Dropout layer with rate = 0.5 after 1st fully connected layer to avoid overfitting
            nn.Dropout(0.5),
            # Normalize bounding box width and height by the image width and height so that they fall between 0 and 1
            nn.Linear(in_features=4096, out_features=7*7*30),  # 7x7x30
            nn.LeakyReLU(0.1),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Final output: 7x7x30 tensor of predictions

# Object detection: single regression problem: image pixels -> bounding box coordinates, class probabilities

# Optimize for sum-squared error
# Increase the loss from bounding box coordinate predictions
lambda_coord = 5 
# Decrease the loss from confidence predictions for boxes that don’t contain objects
lambda_no_obj = 0.5

# Training parameters
epoch = 135
batch_size = 64
momentum = 0.9
decay = 0.0005
lr_1 = 1e-3 # 1 epoch
lr_2 = 1e-2 # 75 epochs
lr_3 = 1e-3 # 30 epochs
lr_4 = 1e-4 # 30 epochs

# Initialize the model, define the loss and optimizer
model = YOLOv1()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_1, momentum=momentum, weight_decay=decay)

# Function for adjusting learning rate
def adjust_learning_rate(optimizer, epoch):
    if epoch < 1:
        lr = lr_1
    elif epoch < 76:
        lr = lr_2
    elif epoch < 106:
        lr = lr_3
    else:
        lr = lr_4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training loop
for epoch in range(epoch):
    adjust_learning_rate(optimizer, epoch)
    model.train()

# define the Detection object
Detection = ["image_path", "gt", "pred"]
def bb_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# Inference
def detect(image, model):
    model.eval()
    with torch.no_grad():
        # Apply the transformation to the input image
        transformed = transform(image=image)
        img = transformed["image"]
        
        # Preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W) format
        img = img.unsqueeze(0)  # Add batch dimension
        
        # Get predictions from the model
        predictions = model(img)
        
        # Process predictions to extract bounding boxes, class probabilities, etc.
        # Add your post-processing code here
        
        return predictions

# Model summary
# summary(model, (3, 448, 448))