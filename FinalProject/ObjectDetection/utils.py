import numpy as np
import json
import cv2
from typing import List, Dict
import matplotlib.pyplot as plt

def convert_bbox_to_dict(bbox: np.ndarray) -> Dict[str, int]:
    """Convert bounding box array to a dictionary format."""
    bbox_semantic_id, x_min, y_min, x_max, y_max, _ = bbox
    return {
        "class_id": int(bbox_semantic_id),
        "xmin": int(x_min),
        "ymin": int(y_min),
        "xmax": int(x_max),
        "ymax": int(y_max)
    }

def convert_to_bmw_format(numpy_file: str, labels_json_file: str, output_file: str):
    """Convert numpy bounding box data to JSON format with class names."""
    data = np.load(numpy_file)
    with open(labels_json_file, 'r') as f:
        class_id_mapping = json.load(f)

    bounding_boxes = []
    for bbox in data:
        bbox_dict = convert_bbox_to_dict(bbox)
        class_name = class_id_mapping.get(str(bbox_dict["class_id"]), {}).get('class', 'unknown')

        bounding_boxes.append({
            "ObjectClassName": class_name,
            "ObjectClassId": bbox_dict["class_id"],
            "Left": bbox_dict["xmin"],
            "Top": bbox_dict["ymin"],
            "Right": bbox_dict["xmax"],
            "Bottom": bbox_dict["ymax"]
        })

    with open(output_file, 'w') as f:
        json.dump(bounding_boxes, f, indent=4)

def visualize_bounding_boxes(json_file: str, image_file: str) -> np.ndarray:
    """Visualize bounding boxes on an image using data from a JSON file and return the image."""
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Load image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    for bbox in data:
        x_min, y_min = bbox['Left'], bbox['Bottom']
        x_max, y_max = bbox['Right'], bbox['Top']
        class_name = bbox['ObjectClassName']

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # Display class name
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image