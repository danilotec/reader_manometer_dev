import torch
import cv2
from ultralytics import YOLO #type: ignore
from model import AngleRegressor

yolo = YOLO("runs/detect/train/weights/best.pt")
reg = AngleRegressor()
reg.load_state_dict(torch.load("regressor.pt"))
reg.eval()

img = cv2.imread("dataset/regression_dataset/images/image_3.jpg")
result = yolo(img)[0] #type: ignore

for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2] #type: ignore
    crop = cv2.resize(crop, (224, 224))

    t = torch.tensor(crop).float().permute(2, 0, 1).unsqueeze(0) / 255

    ang_norm = reg(t).item()
    angle = ang_norm * 360
    percentual = (angle - 135) / (405 - 135)

    # value = (angle / 360) * 100
    print('Valor', percentual)
    print("Ângulo:", angle)
