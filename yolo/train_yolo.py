from ultralytics import YOLO #type: ignore

model = YOLO("yolov8n.pt")
model.train(
data="yolo/gauge.yaml",
epochs=100,
imgsz=640,
batch=16
)
