import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from UNETmodel import CustomModel  # Assuming this is your model architecture
from utils import conv_prediction_img
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths
CHECKPOINT_PATH = "SemanticSegmentation/models/best.pt"
IMAGE_PATH = 'path/to/your/new/image.jpg'  # Update with your new image path

# Image dimensions
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# Load the model
model = CustomModel(in_channels=3, out_channels=10).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)  # Update for safety
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Define transformation for the image
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

# Visualization function
def visualize_prediction(image, mask, output):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # Adjust figsize as needed

    # Display original image
    axes[0].imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display ground truth mask if available
    if mask is not None:
        axes[1].imshow(mask)  # Ground truth mask
        axes[1].set_title("Mask")
    else:
        axes[1].imshow(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))  # Placeholder if no mask
        axes[1].set_title("Mask (N/A)")
    axes[1].axis('off')

    # Display predicted mask
    pred_class = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (720, 1280)
    axes[2].imshow(pred_class)  # Predicted mask
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.show()

# Main function to run inference on a new image
def run_inference_on_new_image():
    try:
        # Load and transform the new image
        image = Image.open(IMAGE_PATH).convert("RGB")
        transformed_image = transform_image(np.array(image))  # Convert PIL image to numpy array

        # Add batch dimension
        transformed_image = transformed_image.unsqueeze(0)  # [1, 3, 720, 1280]

        # Run prediction
        output = predict(transformed_image, model)

        # Save the prediction
        output_path = "new_image_prediction.png"
        save_prediction(output, output_path)

        # Display the images
        visualize_prediction(image, None, output)  # Pass None for mask if not available

    except Exception as e:
        print(f"Error during image processing or inference: {e}")

if __name__ == "__main__":
    run_inference_on_new_image()
