import json
import os
import cv2
import asyncio
from ultralytics import YOLO
from datetime import datetime
import easyocr

model = YOLO("best.pt")

CLASS_NAMES = {0: "with-helmet", 1: "without-helmet", 2: "rider", 3: "number_plate"}  

VIOLATION_FILE = "violations.json"

violations = []

reader = easyocr.Reader(['en'])

async def process_rider(image_path, img, rider_bbox):
    global violations

    x1, y1, x2, y2 = rider_bbox
    rider_crop = img[y1:y2, x1:x2]
    cv2.imwrite("cropped.jpg",rider_crop)
    results = model(rider_crop)

    has_helmet = False
    has_no_helmet = False
    number_plate_bbox = None

    for result in results:
        result.show()
        for box in result.boxes:
            cls = int(box.cls[0])
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])

            if CLASS_NAMES[cls] == "with-helmet":
                has_helmet = True
            elif CLASS_NAMES[cls] == "without-helmet":
                has_no_helmet = True
            elif CLASS_NAMES[cls] == "number_plate":
                number_plate_bbox = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
                print(get_text(img[number_plate_bbox[1]:number_plate_bbox[3], number_plate_bbox[0]:number_plate_bbox[2]]))

    if has_no_helmet and not has_helmet:
        print("here")
        # print()
        violation_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": os.path.basename(image_path),
            "rider_bbox": rider_bbox,
            "number_plate": get_text(img[number_plate_bbox[1]:number_plate_bbox[3], number_plate_bbox[0]:number_plate_bbox[2]])
        }
        violations = json.load(open("violations.json",'r'))
        violations.append(violation_entry)
        print(violations)
        with open("violations.json","w") as file:
            json.dump(violations,file,indent=4)
        violations.append(violation_entry)

def get_text(frame):
    results = reader.readtext(frame)
    plate_text = " ".join([res[1] for res in results])
    return plate_text if plate_text else "N/A"

async def detect_violations(image_path):
    img = cv2.imread(image_path)
    results = model(image_path)

    tasks = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if CLASS_NAMES[cls] == "rider":
                tasks.append(process_rider(image_path, img, [x1, y1, x2, y2]))

    await asyncio.gather(*tasks)

async def main():
    # upload and add image path here
    await detect_violations("d5.jpg")

    print(f"Violations saved to {VIOLATION_FILE}")


asyncio.run(main())
