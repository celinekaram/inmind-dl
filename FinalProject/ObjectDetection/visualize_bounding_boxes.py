import os
import cv2
from torch.utils.data import DataLoader
from dataset import BMWObjectDataset
from utils import convert_to_bmw_format, visualize_bounding_boxes

# Paths to directories
npy_dir = 'organized_data/bounding_boxes/tight_npy'
bb_data_dir = 'organized_data/bounding_boxes/bb_data_json'
tight_labels_dir = 'organized_data/tight_labels_json'
img_dir = 'organized_data/rgb_images'
labeled_img_dir = 'organized_data/bounding_boxes/labeled_img'

# Create output directories if they don't exist
os.makedirs(bb_data_dir, exist_ok=True)
os.makedirs(labeled_img_dir, exist_ok=True)

def main():
    # Initialize dataset and dataloader
    dataset = BMWObjectDataset(img_dir, npy_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over the dataset
    for idx, (image, _) in enumerate(dataloader):
        # Define output paths for JSON files and images
        json_output_path = os.path.join(bb_data_dir, f'{idx:04d}.json')
        img_output_path = os.path.join(labeled_img_dir, f'{idx:04d}.png')
        
        # Convert bounding boxes and save as JSON
        convert_to_bmw_format(
            os.path.join(npy_dir, f'{idx:04d}.npy'),
            os.path.join(tight_labels_dir, f'{idx:04d}.json'),
            json_output_path
        )
        
        # Visualize bounding boxes and return the image
        visualized_image = visualize_bounding_boxes(json_output_path, os.path.join(img_dir, f'{idx:04d}.png'))
        
        # Save the visualized image
        cv2.imwrite(img_output_path, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
        
if __name__ == "__main__":
    main()