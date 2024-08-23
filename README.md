## README

### Inmind Academy Code Collection

**Author:** Celine Karam

**Note:** This repository provides code snippets and examples for educational purposes. It might not be a complete, standalone project.

This repository contains a collection of code snippets and examples created during the Inmind Academy Machine Learning Track.

### FinalProject

git clone https://github.com/celinekaram/inmind-dl
cd FinalProject
python organize_dataset.py
# Object Detection
cd ObjectDetection
python ObjectDetection/yolo_converter.py
python ObjectDetection/split.py
# Train
cd yolov5
python train.py --data data.yaml --weights runs/train/exp1/weights/best.pt --img 640 --epochs 1 --batch-size 2
# Inference
python detect.py --weights runs/train/exp1/weights/best.pt --source ../data/test/images
# ONNX
pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
# Tensorboard
tensorboard --logdir logs
# Netron
netron ObjectDetection/yolov5/runs/train/exp4/weights/best.pt
netron SemanticSegmentation/models/bmw.onnx
# Train semantic segmentation (use Trining.ipynb)