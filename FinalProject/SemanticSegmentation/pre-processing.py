# File to visualize all class names and colors across entire dataset
import os
import json

# Directory containing the JSON files
dir = "organized_data/semantic_segmentation/semantic_label_json"

# Initialize an empty dictionary to store unique class-color combinations
class_color_map = {}

# Iterate through all files in the specified directory
for file_name in os.listdir(dir):
    file_path = os.path.join(dir, file_name)
    
    # Load the JSON file
    with open(file_path) as f:
        data = json.load(f)
        
        # Iterate through each color-class pair in the JSON data
        for color, class_dict in data.items():
            class_name = class_dict['class']
            
            # Check if the class_name is already in the class_color_map
            if class_name not in class_color_map:
                # If not, add it with the corresponding color
                class_color_map[class_name] = color
            elif color != class_color_map[class_name]:
                # If the class_name already exists, but with a different color, print a warning
                print(f"Class name: {class_name} had color: {class_color_map[class_name]}, "
                      f"but a different color {color} was found in {file_name}.json")

# Print the final class-color map
print(class_color_map)