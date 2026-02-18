from ultralytics import YOLO
import os

def main():
    os.chdir('e:/Dota_yolo_project')

    # Load YOLOv8s model (downloads from internet on first run)
    print("Loading YOLOv8s model...")
    model = YOLO('yolov8s.pt')

    # Train (using GPU)
    print("Starting training on GPU...")
    results = model.train(
        data='dataset/data.yaml',
        imgsz=1024,
        epochs=100,
        batch=4,
        device=0,
        patience=20,
        save=True,
        project='runs/detect',
        name='train',
        verbose=True,
        workers=0  # Disable multiprocessing workers to avoid Windows issues
    )

    print("Training complete!")

if __name__ == '__main__':
    main()
