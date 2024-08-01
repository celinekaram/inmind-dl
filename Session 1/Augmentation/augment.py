import albumentations as A
import cv2
import os

# Define paths
images_path = "./images"
labels_path = "./labels"
output_images_path = "./augmented_images"
output_labels_path = "./augmented_labels"

# Create output directories if they don't exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

def augment_and_save(image_path, label_path, output_image_path, output_label_path):
    # Load the image
    image = cv2.imread(image_path)

    # Open the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store bounding boxes
    bboxes = []

    # Iterate through each line
    for line in lines:
        # Split the line by whitespace to get class and coordinates
        data = line.strip().split()
        # Extract the coordinates and convert them to floats
        # class x_center y_center width height
        bbox = [float(coord) for coord in data[1:]]
        bbox.append(data[0])
        # Append the bounding box to the list of bounding boxes
        bboxes.append(bbox)

    # Create an instance of the Albumentations Compose object containing a list of augmentation transformations to apply
    transform = A.Compose([
        # Mirror % y with a probability 50%
        A.HorizontalFlip(p=0.5), 
        # Change brightness and contrast of the image randomly with a 20% probability
        A.RandomBrightnessContrast(p=0.2), 
        # Rotate image randomly within the range of -130 to 130 degrees with a 50% probability
        A.Rotate(limit=130, p=0.5), 
        # Apply blur effect to image with a maximum kernel size of 155, with a 20% probability
        A.Blur(blur_limit=155, p=0.2), 
        # Add Gaussian noise to the image with variance in the range of 10 to 50, with a 20% probability
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        # Randomly shift, scale, and rotate the image within the given limits, with a 50% probability
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),  
        # Randomly crop and resize the image to 480x480 pixels
        # A.RandomResizedCrop(480, 480),  
        # Randomly shift the RGB channels within the given limits, with a 50% probability
        A.RGBShift(r_shift_limit=200, g_shift_limit=200, b_shift_limit=200, p=0.5), 
        # Randomly change the brightness of the image within the limit of 20%, with a 50% probability 
        # A.RandomBrightness(limit=0.2, p=0.5),  
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to image with a clip limit of 4.0, with a 50% probability
        A.CLAHE(clip_limit=4.0, p=0.5),  
    ], bbox_params=A.BboxParams(format='yolo'))

    # Apply the transformation
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    # Convert the bounding boxes from relative to absolute coordinates
    height, width, _ = transformed_image.shape

    # Draw bounding boxes on the original image
    for bbox in bboxes:
        x_center, y_center, width, height, _ = bbox
        img_height, img_width = image.shape[:2]
        xmin = int((x_center - width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        xmax = int((x_center + width / 2) * img_width)
        ymax = int((y_center + height / 2) * img_height)
        
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

    # Draw bounding boxes on the augmented image
    ## Tuple to list
    absolute_bboxes = [[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]] for bbox in transformed_bboxes]

    for bbox in absolute_bboxes:
        x_center, y_center, width, height, _ = bbox
        xmin = int((x_center - width / 2)* img_width)
        ymin = int((y_center - height / 2)* img_height)
        xmax = int((x_center + width / 2)* img_width)
        ymax = int((y_center + height / 2)* img_height)
        cv2.rectangle(transformed_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        
        # Save the transformed image
    cv2.imwrite(output_image_path, transformed_image)

    # Save the transformed labels
    with open(output_label_path, 'w') as file:
        for bbox in transformed_bboxes:
            class_label = bbox[4]
            bbox_str = ' '.join(map(str, bbox[:4]))
            file.write(f"{class_label} {bbox_str}\n")

# Loop through the dataset
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(images_path, filename)
        label_path = os.path.join(labels_path, filename.replace(".jpg", ".txt"))
        
        output_image_path = os.path.join(output_images_path, filename)
        output_label_path = os.path.join(output_labels_path, filename.replace(".jpg", ".txt"))
        
        augment_and_save(image_path, label_path, output_image_path, output_label_path)