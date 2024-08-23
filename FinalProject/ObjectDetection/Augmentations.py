
import albumentations as A
import os
import cv2

transform = A.Compose(
    [
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75
        ),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

imagesTransformed = []
    
def apply_transformations(bounding_boxes_data, transform, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for idx, data in enumerate(bounding_boxes_data):
        try:
            img_file = data["image_file"]
            img_path = os.path.join("./data", img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            bboxes = data["bounding_box_data"]

            class_labels = [int(x[0]) for x in bboxes]

            # Convert bboxes to YOLO format
            height, width = img.shape[:2]
            yolo_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox[1], bbox[2], bbox[3], bbox[4]
                x_center = (x_min + x_max) / (2 * width)
                y_center = (y_min + y_max) / (2 * height)
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height
                yolo_bboxes.append([x_center, y_center, box_width, box_height])

            # Apply all transformations
            for trans_idx, trans in enumerate(transform):
                transformed = trans(
                    image=img, bboxes=yolo_bboxes, class_labels=class_labels
                )
                transformed_image = transformed["image"]
                transformed_bboxes = transformed["bboxes"]
                transformed_labels = transformed["class_labels"]

                # Save transformed image
                output_image_file = f"{os.path.splitext(img_file)[0]}_{trans_idx}{os.path.splitext(img_file)[1]}"
                output_path = os.path.join(output_folder, output_image_file)
                cv2.imwrite(output_path, transformed_image)

                # Save transformed labels
                label_file = f"{os.path.splitext(img_file)[0]}_{trans_idx}.txt"
                label_path = os.path.join(output_folder, label_file)

                with open(label_path, "w") as file:
                    for label, bbox in zip(transformed_labels, transformed_bboxes):
                        file.write(f"{label} {' '.join(map(str, bbox))}\n")

            print(f"Processed image {idx + 1}/{len(bounding_boxes_data)}: {img_file}")

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue


# Usage
output_folder = "./transformedData"
apply_transformations(bounding_boxes_data, transform, output_folder)