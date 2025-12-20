import torch
import cv2
from model import AngleRegressor

model = AngleRegressor()
model.load_state_dict(torch.load("regressor.pt"))
model.eval()

img = cv2.imread("dataset/regression_dataset/images/image_1.jpg")
img = cv2.resize(img, (224, 224)) #type: ignore
img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255

pred = model(img).item()
print("Ângulo previsto:", pred * 360)
