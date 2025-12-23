import torch
import cv2
from ultralytics import YOLO #type: ignore
from .model import AngleRegressor

class Manometer:
    def __init__(self, model: str) -> None: 
        self.yolo = YOLO(model)
        self.reg = AngleRegressor()
        self.reg.load_state_dict(torch.load("regressor.pt"))
        self.reg.eval()

    def get_angle(self, filename: str) -> list | None:
        img = cv2.imread(filename)
        result = self.yolo(img)[0] #type: ignore

        angles = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2] #type: ignore
            crop = cv2.resize(crop, (224, 224))

            t = torch.tensor(crop).float().permute(2, 0, 1).unsqueeze(0) / 255

            ang_norm = self.reg(t).item()
            angles.append(ang_norm * 360)
        return angles