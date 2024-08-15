import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import convert_bbox_to_dict

class BMWObjectDataset(Dataset):
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
        bboxes = [convert_bbox_to_dict(bbox) for bbox in bboxes]

        return image, bboxes