import albumentations as A
import cv2

# Read image
image = cv2.imread("./images/000011.jpg")
#from matplotlib.image import imread
#image = imread(image_path)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Open the text file
with open('./labels/000011.txt', 'r') as file:
    # Read the lines of the file
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

# Display the result
print(bboxes)


# Apply augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=130, p=0.5),
    A.Blur(blur_limit=300, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    #A.RandomResizedCrop(480, 480),
    A.RGBShift(r_shift_limit=200, g_shift_limit=200, b_shift_limit=200, p=0.5),
    #A.RandomBrightness(limit=0.2, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
], bbox_params=A.BboxParams(format='yolo'))


transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']

print(transformed_bboxes)


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

# Display the original and augmented images with bounding boxes
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
cv2.imshow("Augmented Image", transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
