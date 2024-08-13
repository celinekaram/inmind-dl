import os
import numpy as np
import json
import cv2
from typing import List, Dict
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Image dimensions (these should be the dimensions of the images you're working with)
IMG_WIDTH = 640  # Default image size used for training in Yolov7
IMG_HEIGHT = 640  # Default image size used for training in Yolov7

# Convert Numpy array containing bounding box data to Two-Point Coordinates format
def convert_to_two_point_coordinates(bbox: np.ndarray) -> Dict[str, int]:
    """
    Converts a bounding box numpy array to a dictionary with two-point coordinates.
    
    Args:
        bbox (np.ndarray): Array containing bounding box data.
    
    Returns:
        Dict[str, int]: A dictionary with the bounding box coordinates and class ID.
    """
    bbox_semantic_id, x_min, y_min, x_max, y_max, occ_rate = bbox

    return {
        "class_id": int(bbox_semantic_id),
        "xmin": int(x_min),
        "ymin": int(y_min),
        "xmax": int(x_max),
        "ymax": int(y_max)
    }

# Load numpy data, convert bounding boxes to Two-Point Coordinates format, and save to a JSON file
def create_output_json(file_path: str, output_file_path: str) -> None:
    """
    Loads a .npy file, converts bounding boxes to Two-Point Coordinates format, 
    and saves them to a JSON file.
    
    Args:
        file_path (str): Path to the .npy file containing bounding box data.
        output_file_path (str): Path to the output JSON file.
    """

    # Load the .npy file
    data = np.load(file_path)

    # Convert all bounding boxes to Two-Point Coordinates format
    two_point_data: List[Dict[str, int]] = [convert_to_two_point_coordinates(bbox) for bbox in data]

    # Write to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(two_point_data, f, indent=4)

# Visualize bounding boxes on an image using data from a JSON file
def visualize_bb(json_file: str, image_file: str) -> None:
    """
    Visualizes bounding boxes on an image using data from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing bounding box data.
        image_file (str): Path to the image file.
    """
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Load image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (OpenCV uses BGR by default)

    # Plot image with bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    # Draw bounding boxes
    for bbox in data:
        class_name = bbox.get('class_id')
        x_min, y_min = bbox['xmin'], bbox['ymin']
        x_max, y_max = bbox['xmax'], bbox['ymax']

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Display class ID
        cv2.putText(image, str(class_name), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display image with bounding boxes
    plt.axis('off')  # Turn off axis labels
    plt.imshow(image)
    plt.show()

class CustomDataset(Dataset):
    """
    Custom Dataset class to load images and their corresponding bounding boxes.
    """
    def __init__(self, img_dir: str, bbox_dir: str):
        self.img_dir = img_dir
        self.bbox_dir = bbox_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.bbox_files = sorted(os.listdir(bbox_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        bbox_path = os.path.join(self.bbox_dir, self.bbox_files[idx])

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load bounding boxes
        bboxes = np.load(bbox_path)
        bboxes = [convert_to_two_point_coordinates(bbox) for bbox in bboxes]

        return image, bboxes

# Paths to directories
npy_dir = './.gitignore/organized_data/bounding_boxes/tight'
output_dir = './.gitignore/organized_data/bounding_boxes/output'
img_dir = './.gitignore/organized_data/rgb_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize dataset and dataloader
dataset = CustomDataset(img_dir, npy_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the dataset
for idx, (image, bboxes) in enumerate(dataloader):
    # Define output paths for JSON files
    output_file_path = os.path.join(output_dir, f'{idx:04d}.json')
    
    # Convert bounding boxes and save as JSON
    create_output_json(os.path.join(npy_dir, f'{idx:04d}.npy'), output_file_path)
    
    # Visualize the bounding boxes on the image
    visualize_bb(output_file_path, os.path.join(img_dir, f'{idx:04d}.png'))