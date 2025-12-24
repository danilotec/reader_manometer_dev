from ultralytics.models import YOLO
import cv2
import os

class CropImage:
    def __init__(self, yolo: YOLO, imput_dir: str, output_dir: str) -> None:
        
        self.yolo = yolo
        self.input_dir = imput_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_crop(self, img_name: str, classe: int = 1) -> None:
        img = cv2.imread(os.path.join(self.input_dir, img_name))
        
        if img is not None:
            results = self.yolo(img)[0]

            print(img_name)
            for i, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                if cls == classe:  # Needle
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (224, 224))
                    name = img_name.replace(".png", "").replace(".jpg", "").replace('.jpeg', '')
                    cv2.imwrite(f"{self.output_dir}/{name}_{i}.jpg", crop)
        else:
            return None