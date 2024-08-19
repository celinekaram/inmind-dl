import numpy as np
import json
from PIL import Image
import os

# Define number of classes after data preprocessing
num_classes = 10

# Define directories
semantic_label_dir = "organized_data/semantic_segmentation/semantic_label_json"
tight_label_dir = "organized_data/tight_labels_json"

# Create a mapping from class_id to color tuple
def create_class_id_to_color_map(idx):
    # Construct file paths correctly
    semantic_label_path = os.path.join(semantic_label_dir, f'{idx}.json')
    tight_label_path = os.path.join(tight_label_dir, f'{idx}.json')

    # Load JSON files
    with open(semantic_label_path) as f:
        semantic_label = json.load(f)
    with open(tight_label_path) as f:
        tight_label = json.load(f)
    
    # Create the mapping from class_id to color
    class_id_to_color_map = {}
    for color_str, class_name_semantic in semantic_label.items():
        # Convert the color string to an RGBA tuple
        color_tuple = tuple(map(int, color_str.strip('()').split(',')))
        rgb_color = color_tuple[:3]  # take first 3 channels: rgb
        for class_id, class_name_tight in tight_label.items():
            if class_name_tight == class_name_semantic:
                class_id_to_color_map[int(class_id)] = rgb_color
                
    return class_id_to_color_map

def color_to_class_and_one_hot(mask, idx, num_classes=num_classes):
    """
    Converts an RGB mask to a class index mask and then to a one-hot encoded mask.

    Args:
    - mask (numpy array): The input RGB mask image.
    - idx (int): The index of input.
    - num_classes (int): The total number of classes.

    Returns:
    - one_hot_mask (numpy array): The one-hot encoded mask with shape (height, width, num_classes).
    """
    class_id_to_color_map = create_class_id_to_color_map(idx)
    
    # Get the height and width of the mask
    height, width, _ = mask.shape  
    
    # Initialize the mask output with zeros, where each pixel corresponds to a class ID
    mask_output = np.zeros((height, width, num_classes), dtype=np.uint8)

    # Iterate over the class_id to color mapping
    for i in range (height):
        for j in range (width):
            color_tuple = tuple(mask[i, j])
            for class_id, color in class_id_to_color_map.items():
                if color_tuple == color:
                    print('yes')
                    mask_id = class_id
                else:
                    mask_id = num_classes - 1   # last class for unlabeled/ background/ other
            mask_output[i, j, mask_id] = 1 # one hot encoding

    return mask_output

# Create the class_id to color mapping
# class_id_to_color_map = create_class_id_to_color_map('0002')
# print(class_id_to_color_map)

# Load the mask image
mask_path = 'organized_data/semantic_segmentation/semantic_image/0002.png'
mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)

# Convert the mask to one-hot encoding
one_hot_mask = color_to_class_and_one_hot(mask, '0002')