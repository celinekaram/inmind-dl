import torch
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from UNETmodel import CustomModel

# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = CustomModel(3, 10)

# Load the checkpoint
checkpoint = torch.load("SemanticSegmentation/models/best.pt", map_location=device)

# Extract the model's state_dict from the checkpoint
model_state_dict = checkpoint['state_dict']

# Load the state_dict into the model
model.load_state_dict(model_state_dict)

# Move the model to the appropriate device (CPU or GPU)
model.to(device)

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor with the desired shape (1, 3, 1280, 1280)
dummy_input = torch.randn(1, 3, 1280, 1280).to(device)

# Define the path to save the ONNX model
onnx_path = "SemanticSegmentation/models/bmw.onnx"

# Export the PyTorch model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    opset_version=11
)

# Load and check the ONNX model for correctness
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Create an inference session using ONNX Runtime
ort_sess = ort.InferenceSession(onnx_path)

# Load an image and apply necessary transformations for model input
transform = transforms.Compose([
    transforms.Resize((1280, 1280)),  # Resize image to match model input size
    transforms.ToTensor()             # Convert image to tensor
])

image_path = 'SemanticSegmentation/data/test_small/images/0002.png'
image = Image.open(image_path).convert('RGB')
x = transform(image).unsqueeze(0).numpy()  # Add batch dimension and convert to numpy array
x = x.astype(np.float32)  # Ensure the image is in float32 format for ONNX input

# Run inference with PyTorch
start_time = time.time()
with torch.no_grad():
    torch_output = model(torch.tensor(x).to(device)).cpu().numpy()
torch_inference_time = time.time() - start_time

# Run inference with ONNX Runtime
start_time = time.time()
onnx_outputs = ort_sess.run(None, {'input': x})
onnx_inference_time = time.time() - start_time

# Print inference times for comparison
print(f"PyTorch inference time: {torch_inference_time:.4f} seconds")
print(f"ONNX Runtime inference time: {onnx_inference_time:.4f} seconds")

id_color_map = {
    0: (255, 197, 25),
    1: (140, 255, 25),
    2: (140, 25, 255),
    3: (226, 255, 25),
    4: (255, 111, 25),
    5: (255, 25, 197),
    6: (54, 255, 25),
    7: (25, 255, 82),
    8: (25, 82, 255),
    9: (0, 0, 0)
}

def conv_prediction_img(mask_prediction, id_color_map=id_color_map):
    max_class = torch.argmax(mask_prediction, dim=1)  # Shape [batch_size, height, width]
    batch_size, height, width = max_class.shape
    output_image = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)

    for class_idx, color in id_color_map.items():
        # Create a mask where the class index is equal to class_idx
        mask = (max_class == class_idx).unsqueeze(1)  # Shape [batch_size, 1, height, width]
        color_tensor = torch.tensor(color, dtype=torch.uint8).view(1, 3, 1, 1)  # Shape [1, 3, 1, 1]
        # Assign the color to the output image where the mask is True
        output_image[mask] = color_tensor.expand_as(output_image)[mask]

    return output_image

# Process and print the results
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
outputs = model(image_tensor)
outputsRgb = conv_prediction_img(torch.tensor(outputs).to(device))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0))
axes[0].set_title("Image")

axes[1].imshow(outputsRgb.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
axes[1].set_title("Mask")

plt.show()

# Optional: Save or display the output image if needed
cv2.imwrite('output_image_torch.png', outputs[0].argmax(0).cpu().numpy())
cv2.imwrite('output_image_onnx.png', onnx_outputs[0][0].argmax(0))