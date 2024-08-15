import os
import shutil

# Define the base directory where dataset is located
base_dir = 'raw_data/'
new_dir = 'organized_data/'

# Create new directory if they don't exist
os.makedirs(new_dir, exist_ok=True)

# Define the new directories where the files will be moved
directories = {
    # RGB image: original colored images from the dataset
    "rgb_image": os.path.join(new_dir, 'rgb_images/'),
    # Semantic image: showing different objects in the scene with distinct colors
    "semantic_image": os.path.join(new_dir, 'semantic_segmentation/semantic_image/'),
    # Bounding box numpy: bounding box coordinates
    "bounding_box_npy": os.path.join(new_dir, 'bounding_boxes/tight_npy/'),
    # Bounding box label: mapping class_id to class_name for the bounding boxes.
    "bounding_box_labels": os.path.join(new_dir, 'tight_labels_json/'),
    # Bounding box primary path: primitive paths corresponding to objects in the scene
    "bounding_box_prim_paths": os.path.join(new_dir, 'tight_prim_paths_json/'),
    # Semantic labels: JSON files mapping colors to class names in the semantic segmentation images
    "semantic_labels": os.path.join(new_dir, 'semantic_segmentation/semantic_label_json/')
}

# Create directories if they don't exist
for key in directories:
    if not os.path.exists(directories[key]):
        os.makedirs(directories[key])

# Function to assign and rename files
def organize_files(base_dir, directories):
    for filename in os.listdir(base_dir):
        if filename.startswith('rgb_'):
            new_filename = filename.replace('rgb_', '')  # Extracting the ID
            shutil.copy(os.path.join(base_dir, filename), os.path.join(directories["rgb_image"], new_filename))
        
        elif filename.startswith('semantic_segmentation_') and filename.endswith('.png'):
            new_filename = filename.replace('semantic_segmentation_', '')  # Extracting the ID
            shutil.copy(os.path.join(base_dir, filename), os.path.join(directories["semantic_image"], new_filename))
        
        elif filename.startswith('bounding_box_2d_tight_') and filename.endswith('.npy'):
            new_filename = filename.replace('bounding_box_2d_tight_', '')  # Extracting the ID
            shutil.copy(os.path.join(base_dir, filename), os.path.join(directories["bounding_box_npy"], new_filename))
        
        elif filename.startswith('bounding_box_2d_tight_labels_') and filename.endswith('.json'):
            new_filename = filename.replace('bounding_box_2d_tight_labels_', '')  # Extracting the ID
            shutil.copy(os.path.join(base_dir, filename), os.path.join(directories["bounding_box_labels"], new_filename))
        
        elif filename.startswith('bounding_box_2d_tight_prim_paths_') and filename.endswith('.json'):
            new_filename = filename.replace('bounding_box_2d_tight_prim_paths_', '')  # Extracting the ID
            shutil.copy(os.path.join(base_dir, filename), os.path.join(directories["bounding_box_prim_paths"], new_filename))
        
        elif filename.startswith('semantic_segmentation_labels_') and filename.endswith('.json'):
            new_filename = filename.replace('semantic_segmentation_labels_', '')  # Extracting the ID
            shutil.copy(os.path.join(base_dir, filename), os.path.join(directories["semantic_labels"], new_filename))

# Run the function
organize_files(base_dir, directories)

print("Files have been successfully copied, organized, and renamed.")