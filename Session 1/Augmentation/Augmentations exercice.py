import albumentations as A
import cv2

# Load the 1st image
image = cv2.imread("./images/000001.jpg")
# image = cv2.imread("./images/000001.jpg", cv2.IMREAD_GRAYSCALE)

# Display the 1st image
cv2.imshow("Image", image)

# Wait for the user to press a key
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()

# Open the 1st label
with open('./labels/000001.txt', 'r') as file:
    # Read the lines of the file
    lines = file.readlines()

print(lines)

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
