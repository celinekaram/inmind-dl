import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNET  # Make sure this is the correct import for your model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Hyperparameters
DEVICE = "cpu"  # or "cuda" if you're using a GPU
MODEL_PATH = 'Session3/Carvana/model_2.pth.tar'
IMAGE_PATH = 'C:/Datasets/Carvana/data/test_image/test1.jpg'
OUTPUT_PATH = 'output_mask.gif'

# Define transformations (must match the training transforms)
transform = A.Compose([
    A.Resize(height=480, width=720),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
    ToTensorV2()
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB if needed
    return image

def preprocess_image(image):
    image = np.array(image)  # Convert PIL Image to numpy array
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0)  # Add batch dimension

def postprocess_prediction(prediction):
    # Convert tensor to numpy array and squeeze the batch dimension
    prediction = prediction.squeeze().cpu().numpy()
    # Apply threshold if needed (e.g., for binary segmentation)
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

def save_mask(mask, output_path):
    mask_image = Image.fromarray(mask)
    mask_image.save(output_path)

def main():
    # Load the model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['state_dict'])
    model.eval()
    
    # Load and preprocess the image
    image = load_image(IMAGE_PATH)
    input_tensor = preprocess_image(image).to(DEVICE)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process the output
    mask = postprocess_prediction(output)
    
    # Save or display the result
    save_mask(mask, OUTPUT_PATH)
    print(f"Mask saved to {OUTPUT_PATH}")
    
    # Optionally, display the mask
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()