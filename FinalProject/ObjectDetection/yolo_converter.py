import numpy as np
import os
from typing import Dict, List

def convert_to_yolo_format(bbox: np.ndarray, img_width: int, img_height: int) -> List[float]:
    """
    Convert bounding box coordinates to YOLO format.
    
    Args:
        bbox (np.ndarray): Bounding box data [bbox_semantic_id, x_min, y_min, x_max, y_max, occ_rate].
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        List[float]: List with class_id, x_center, y_center, width, height in YOLO format.
    """
    bbox_semantic_id, x_min, y_min, x_max, y_max, _ = bbox

    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return [int(bbox_semantic_id), x_center, y_center, width, height]

def main():
    
    # Original Image dimensions
    img_width = 1280
    img_height = 720
    
    # Define paths
    npy_dir = 'organized_data/bounding_boxes/tight_npy'
    bb_yolo_format_dir = 'organized_data/bounding_boxes/bb_yolo_format'
    
    # Create output directories if they don't exist
    os.makedirs(bb_yolo_format_dir, exist_ok=True)

    """
    Convert bounding boxes from .npy files to YOLO format and save to .txt files.

    Args:
        npy_dir (str): Directory containing the .npy files.
        output_dir (str): Directory where YOLO .txt files will be saved.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
    """
    os.makedirs(bb_yolo_format_dir, exist_ok=True)

    for file_name in sorted(os.listdir(npy_dir)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(npy_dir, file_name)
            data = np.load(file_path)

            # Convert each bounding box to YOLO format
            yolo_data = [convert_to_yolo_format(bbox, img_width, img_height) for bbox in data]

            # Define the output path for the YOLO .txt file
            output_file_path = os.path.join(bb_yolo_format_dir, file_name.replace('.npy', '.txt'))

            # Write the YOLO data to a .txt file
            with open(output_file_path, 'w') as f:
                for bbox in yolo_data:
                    bbox_str = ' '.join(map(str, bbox))
                    f.write(f"{bbox_str}\n")

    print(f"YOLO formatted data saved to {bb_yolo_format_dir}!")

if __name__ == "__main__":
    main()