import torch
import cv2
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from model import AngleRegressor


class NeedleDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image"])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224)) #type: ignore
        img = torch.tensor(img).float().permute(2, 0, 1) / 255.0

        angle = row["angle"] / 360.0  # normaliza
        angle = torch.tensor([angle]).float()

        return img, angle



device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = NeedleDataset(
    csv_file="dataset/regression_dataset/labels.csv",
    img_dir="dataset/regression_dataset/images"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AngleRegressor().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


epochs = 50

for epoch in range(epochs):
    total_loss = 0

    for imgs, angles in loader:
        imgs = imgs.to(device)
        angles = angles.to(device)

        preds = model(imgs)
        loss = criterion(preds, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.5f}")


torch.save(model.state_dict(), "regressor.pt")
print("✅ regressor.pt salvo com sucesso")
