#!/usr/bin/python3
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom

# Images
img = 'output.avi'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
