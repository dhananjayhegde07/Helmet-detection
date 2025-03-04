import cv2
import torch
import asyncio
import easyocr
import json
import os
from datetime import datetime
from ultralytics import YOLO

# Load YOLO models
helmet_model = YOLO("cur_best.pt")  # Helmet detection model
plate_reader = easyocr.Reader(['en'])  # OCR for number plates

VIOLATION_FILE = "violations.json"  # JSON file for storing violations

def process_cropped_frame(frame):
    """
    Processes the cropped frame:
    - Detects helmet or no helmet.
    - If no helmet, extracts the license plate and saves details.
    """
    # Run helmet detection
    results = helmet_model(frame)
    
    # Check for detected objects
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Get class ID

            if cls == 0:  # No Helmet
                print("🚨 Rider without helmet detected!")

                # Extract and process number plate
                plate_text = extract_plate_text(frame)

                if plate_text:
                    print(f"📌 Detected License Plate: {plate_text}")
                    save_violation(plate_text)
                print("number plate cannot be found")
                return  # Stop processing if a violation is found

    print("✅ Rider has a helmet, ignoring image.")


def extract_plate_text(plate_image):
    """
    Extracts text from the cropped license plate using OCR.
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = plate_reader.readtext(thresh, detail=0)  # Extract text
    return text[0] if text else None

def save_violation(plate_text):
    """
    Saves the violation details (license plate & timestamp) to a JSON file.
    """
    violation_data = {
        "license_plate": plate_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Read existing data
    if os.path.exists(VIOLATION_FILE):
        with open(VIOLATION_FILE, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new violation
    data.append(violation_data)

    # Write back to JSON file
    with open(VIOLATION_FILE, "w") as file:
        json.dump(data, file, indent=4)

    print(f"🚔 Violation saved: {violation_data}")
