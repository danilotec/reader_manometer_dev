#type: ignore
from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train/weights/best.pt")

input_dir = "dataset/raw_images"
output_dir = "dataset/regression_dataset/images"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, img_name))
    results = model(img)[0]

    print(img_name)
    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        if cls == 1:  # Needle
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))
            name = img_name.replace(".png", "").replace(".jpg", "")
            cv2.imwrite(f"{output_dir}/{name}_{i}.jpg", crop)
