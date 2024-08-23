import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from UNETmodel import CustomModel  # Assuming this is your model architecture
from dataset import BMWSemanticDataset
from utils import conv_prediction_img
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths
CHECKPOINT_PATH = "SemanticSegmentation/models/best.pt"
TEST_PATH = 'SemanticSegmentation/data/test_small/images'
TEST_MASKS_PATH = 'SemanticSegmentation/data/test_small/masks'

# Image dimensions
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# Load the model
model = CustomModel(in_channels=3, out_channels=10).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Create Dataset and DataLoader
test_ds = BMWSemanticDataset(TEST_PATH, TEST_MASKS_PATH)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)  # Set num_workers to 0 for easier debugging

# Define transformation for image
def transform_image(image):
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    image = transform(image=image)["image"]
    return image

# Original id_color_map
id_color_map = {
    '0': '(255, 197, 25)',   # golden yellow
    '1': '(140, 255, 25)',   # neon green
    '2': '(140, 25, 255)',   # purple
    '3': '(226, 255, 25)',   # lemon yellow
    '4': '(255, 111, 25)',   # orange
    '5': '(255, 25, 197)',   # fuschia
    '6': '(54, 255, 25)',    # bright green
    '7': '(25, 255, 82)',    # middle sea green
    '8': '(25, 82, 255)',    # azure
    '9': '(0, 0, 0)'         # background, other
}

# Convert string to tuple for id_color_map
def convert_color_string_to_tuple(color_string):
    color_tuple = tuple(map(int, color_string.strip('()').split(',')))
    return color_tuple

# Convert all color strings in the map
id_color_map_converted = {k: convert_color_string_to_tuple(v) for k, v in id_color_map.items()}

# Prediction function
def predict(image, model):
    image = image.to(DEVICE).float() / 255.0
    image = image.permute(0, 3, 1, 2)  # [batch_size, height, width, channels] to [batch_size, channels, height, width]
    with torch.no_grad():
        output = model(image)
    return output

# Save the prediction as an image
def save_prediction(output, output_path):
    # Get the class with the highest probability for each pixel
    pred_class = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (720, 1280)
    
    # Initialize an empty RGB image
    pred_rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    
    # Convert the predicted classes to RGB colors
    for class_id, color in id_color_map_converted.items():
        pred_rgb[pred_class == int(class_id)] = color
    
    # Save the RGB image
    cv2.imwrite(output_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

# Inference on the test data
def run_inference():
    try:
        for i, (image, mask) in enumerate(test_loader):
            outputs = predict(image, model)
            # Print shapes for debugging
            print(image.shape)  # torch.Size([1, 720, 1280, 3])
            print(mask.shape)  # torch.Size([1, 720, 1280, 10])
            print(f"Output: {outputs.shape}")  # torch.Size([1, 10, 720, 1280])
            
            output_path = f"output_mask_{i}.png"  # Save each output with an index
            save_prediction(outputs, output_path)

            # Visualization
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # Adjust figsize as needed
            
            # Display ground truth mask
            mask_class = mask[0].cpu().numpy().argmax(axis=-1)  # Convert to class ids (720, 1280)
            axes[0].imshow(mask_class)  # Ground truth mask
            axes[0].set_title("Ground Truth Mask")
            axes[0].axis('off')  # Hide axis
            
            # Display predicted mask
            pred_class = outputs[0].argmax(dim=0).cpu().numpy()  # (720, 1280)
            axes[1].imshow(pred_class)  # Predicted mask
            axes[1].set_title("Predicted Mask")
            axes[1].axis('off')  # Hide axis
            
            plt.show()
            break  # Remove break to process all images
    except Exception as e:
        print(f"Error during data loading or inference: {e}")

if __name__ == "__main__":
    run_inference()