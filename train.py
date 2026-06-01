from ultralytics import YOLO

def main():

    # Load model YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data="dataset/data.yaml",
        epochs=120,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,          # GPU nếu có
        project="runs",
        name="AirDefense_v1",
        save=True,
        plots=True,
        verbose=True
    )

if __name__ == "__main__":
    main()