import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# Import the function to convert color masks to one-hot encoded class masks
from process_mask import color_to_class_and_one_hot

class BMWSemanticDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Initialize the dataset with image and label directories.

        Args:
            img_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing label masks.
            transform (callable, optional): Optional transformation to be applied on both images and masks.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List and sort all image files in the directory
        self.images = sorted(os.listdir(self.img_dir))  
        # List and sort all mask files in the directory
        self.masks = sorted(os.listdir(self.mask_dir))  
    
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding mask at the specified index.

        Args:
            idx (int): Index of the image/mask pair to retrieve.

        Returns:
            image (numpy array): The RGB image as a NumPy array.
            mask (numpy array): The one-hot encoded mask as a NumPy array.
        """
        # Construct the full file path for the image and mask at the specified index
        image_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load the image and convert it to an RGB NumPy array
        image = np.array(Image.open(image_path).convert("RGB"))
        # Load the mask and convert it to an RGB NumPy array
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
        # Convert the mask from color format to a one-hot encoded class mask
        mask = color_to_class_and_one_hot(mask, idx)
        # We can normalize the image by dividing by 255.0, if required

        if self.transform:
            # If a transformation function is provided, apply it to both the image and mask
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # Get the transformed image
            mask = augmented["mask"]  # Get the transformed mask
        
        return image, mask  # Return the image and corresponding mask

def main():
    images_dir = 'organized_data/rgb_images'  # Directory containing RGB images
    masks_dir = 'organized_data/semantic_segmentation/semantic_image'  # Directory containing mask images
    dataset = BMWSemanticDataset(images_dir, masks_dir)  # Create an instance of the dataset

if __name__ == "__main__":
    main()