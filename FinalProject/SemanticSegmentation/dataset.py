import os
from torch.utils.data import Dataset
from PIL import Image

class BMWSemanticDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Initialize the dataset with image and label directories.

        Args:
            img_dir (str): Path to the directory with images.
            mask_dir (str): Path to the directory with label masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(self.img_dir))  # List of image file names
        self.masks = sorted(os.listdir(self.mask_dir))  # List of mask file names
    
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load the image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            # Apply transformations to both image and mask
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask

# Test usage
images_dir = 'organized_data/rgb_images'
masks_dir = 'organized_data/semantic_segmentation/semantic_image'
dataset = BMWSemanticDataset(images_dir, masks_dir)
