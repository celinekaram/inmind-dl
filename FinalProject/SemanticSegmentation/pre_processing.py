import os
import json
import numpy as np

# Define number of classes
num_classes = 10

# Define paths
semantic_label_dir = "organized_data/semantic_segmentation/semantic_label_json"
tight_label_dir = "organized_data/tight_labels_json"

def create_name_color_map(semantic_label_dir=semantic_label_dir):
    """ 
    Creates a map of class names to colors from JSON files in a directory.
    
    Args:
    - semantic_label_dir (str): Directory containing semantic label JSON files.
    
    Returns:
    - dict: A dictionary mapping class names to their associated colors.
    """
    name_color_map = {}

    for file_name in os.listdir(semantic_label_dir):
        file_path = os.path.join(semantic_label_dir, file_name)

        with open(file_path) as f:
            data = json.load(f)
            
            for color, class_dict in data.items():
                rgba_color = tuple(map(int, color.strip("()").split(", "))) # Convert string to tuple
                rgb_color = rgba_color[:3] # Extract the RGB values
                class_name = class_dict['class']
                if class_name not in name_color_map:
                    name_color_map[class_name] = rgb_color
                elif rgb_color != name_color_map[class_name]:
                    print(f"Class name: {class_name} had color: {name_color_map[class_name]}, "
                          f"but a different color {rgb_color} was found in {file_name}")
    
    # Convert the RGB color back to string
    name_color_str = {
        class_name: f"({', '.join(map(str, color))})"
        for class_name, color in name_color_map.items()
    }

    return name_color_str

def create_id_name_map(tight_label_dir=tight_label_dir):
    """ 
    Creates a map of class IDs to class names from JSON files in a directory.
    
    Args:
    - tight_label_dir (str): Directory containing tight label JSON files.
    
    Returns:
    - dict: A dictionary mapping class IDs to their associated class names.
    """
    id_name_map = {}

    for file_name in os.listdir(tight_label_dir):
        file_path = os.path.join(tight_label_dir, file_name)
        
        with open(file_path) as f:
            data = json.load(f)
            
            for class_id, class_dict in data.items():
                class_name = class_dict['class']
                
                if class_id not in id_name_map:
                    id_name_map[class_id] = class_name
                elif class_name != id_name_map[class_id]:
                    print(f"Class ID: {class_id} had name: {id_name_map[class_id]}, "
                          f"but a different name {class_name} was found in {file_name}")

    return id_name_map

def create_id_color_map(semantic_label_dir=semantic_label_dir, tight_label_dir=tight_label_dir):
    """ 
    Maps class IDs to colors by combining the class name-color map and class ID-name map.
    
    Args:
    - semantic_label_dir (str): Directory containing semantic label JSON files.
    - tight_label_dir (str): Directory containing tight label JSON files.
    
    Returns:
    - dict: A dictionary mapping class IDs to colors with class names as comments.
    """
    name_color_map = create_name_color_map(semantic_label_dir)
    id_name_map = create_id_name_map(tight_label_dir)

    id_color_map = {}

    for class_id, class_name in id_name_map.items():
        if class_name in name_color_map:
            id_color_map[class_id] = name_color_map[class_name]
        else:
            print(f"Warning: No color found for class name {class_name} (ID: {class_id})")

    return id_color_map

def conv_color_class(mask, id_color_map, num_classes):
    """
    Converts an RGB mask to a class index mask and then to a one-hot encoded mask.

    Args:
    - mask (numpy array): The input RGB mask image.
    - idx (int): The index of input.
    - num_classes (int): The total number of classes.

    Returns:
    - one_hot_mask (numpy array): The one-hot encoded mask with shape (height, width, num_classes).
    """
    # Get the height and width of the mask
    height, width, _ = mask.shape  # (height, width, rgb) = (720, 1280, 3)
    
    # Initialize the mask output with zeros, where each pixel corresponds to a class ID
    mask_output = np.zeros((height, width, num_classes), dtype=np.uint8)
    mask_id = num_classes - 1   # last class for unlabeled / background / other
    # Iterate over the class_id to color mapping
    for i in range (height):
        for j in range (width):
            color_tuple = tuple(mask[i, j])
            for class_id, color in id_color_map.items():
                if color_tuple == color:
                    mask_id = class_id
                    break
            mask_output[i, j, mask_id] = 1 # one hot encoding

    return mask_output

def main():
    
    name_color_map = create_name_color_map(semantic_label_dir)
    print(f'Name_color_map: {name_color_map}\n')
        
    id_name_map = create_id_name_map(tight_label_dir)
    print(f'Id_name_map: {id_name_map}\n')
    
    id_color_map = create_id_color_map(semantic_label_dir, tight_label_dir)
    print(f'Id_color_map: {id_color_map}\n')

    # Load the mask image
    from PIL import Image
    mask_path = 'organized_data/semantic_segmentation/semantic_image/0020.png'
    mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
    mask_output = conv_color_class(mask, id_color_map, num_classes)
    print(f'Mask_output.shape = {mask_output.shape}') # (height, width, num_classes)
    print(mask_path)

if __name__ == "__main__":
    main()

#  Name_color_map: {
    # 'BACKGROUND': '(0, 0, 0)', 
    # 'rack': '(140, 255, 25)', 
    # 'UNLABELLED': '(0, 0, 0)', 
    # 'crate': '(140, 25, 255)', 
    # 'forklift': '(255, 197, 25)'
    # 'iwhub': '(25, 255, 82)', 
    # 'dolly': '(25, 82, 255)', 
    # 'pallet': '(255, 25, 197)', 
    # 'railing': '(255, 111, 25)', 
    # 'floor': '(226, 255, 25)', 
    # 'stillage': '(54, 255, 25)'}

# Id_name_map: {
    # '0': 'forklift', 
    # '1': 'rack', 
    # '2': 'crate', 
    # '3': 'floor', 
    # '4': 'railing', 
    # '5': 'pallet', 
    # '6': 'stillage', 
    # '7': 'iwhub', 
    # '8': 'dolly'}

# Id_color_map: {
    # '0': '(255, 197, 25)', # golden yellow
    # '1': '(140, 255, 25)', # neon green
    # '2': '(140, 25, 255)', # purple
    # '3': '(226, 255, 25)', # lemon yellow
    # '4': '(255, 111, 25)', # orange
    # '5': '(255, 25, 197)', # fuschia
    # '6': '(54, 255, 25)', # bright green
    # '7': '(25, 255, 82)', #  middle sea green
    # '8': '(25, 82, 255)'} # azure