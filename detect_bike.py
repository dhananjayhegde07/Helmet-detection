import cv2
import torch
import easyocr
from ultralytics import YOLO
import matplotlib.pyplot as plt
import process_helmet
import asyncio

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 model

# Load image
image_path = "d4.jpeg"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Lists to store detected objects
bikes = []
riders = []
plates = []

# Extract detected objects
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])  # Class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

        if cls == 0:  # Person (Rider)
            riders.append((x1, y1, x2, y2))
        elif cls == 3:  # Motorbike
            bikes.append((x1, y1, x2, y2))

        elif cls == 39:  # License Plate (some YOLO models have number plate class)
            plates.append((x1, y1, x2, y2))

print(riders,bikes,plates)

merged_boxes = []

# Function to check IoU (Intersection over Union)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # Calculate intersection
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

for bike in bikes:
    bx1, by1, bx2, by2 = bike  # Bike box

    for rider in riders:
        rx1, ry1, rx2, ry2 = rider  # Rider box

        if iou(bike, rider) > 0.2:  # Loosen the IoU threshold
            bike_center_y = (by1 + by2) // 2  # Middle of bike
            rider_bottom = ry2  # Rider's bottom

            if rider_bottom <= by2 + 30:  # Allow slight tolerance
                merged_x1 = min(bx1, rx1)
                merged_y1 = min(by1, ry1)
                merged_x2 = max(bx2, rx2)
                merged_y2 = max(by2, ry2)

                merged_boxes.append((merged_x1, merged_y1, merged_x2, merged_y2))

print(merged_boxes)
# Draw merged bounding boxes for bike + rider
for (x1, y1, x2, y2) in merged_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red box for bike + rider
    cv2.putText(image, "Bike + Rider", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    process_helmet.process_cropped_frame(image[y1:y2, x1:x2])
    

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

