from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")
results = model.train(data="segmentation.yaml", epochs=50, imgsz=640, device='cuda:2', fliplr=0)