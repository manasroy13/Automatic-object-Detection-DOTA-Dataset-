# Automatic Object Detection on DOTA Dataset using YOLOv8

This project implements object detection on the DOTA (A Large-scale Object Detection in Aerial Images) dataset using YOLOv8.

## 📋 Project Overview

- **Dataset**: DOTA v1.0 (15 classes of objects in aerial images)
- **Model**: YOLOv8 (Ultralytics)
- **Task**: Object detection in aerial/satellite imagery
- **Classes**: 15 types (plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool)

## 🚀 Quick Start

### Prerequisites
```bash
python 3.8+
pip
GPU (CUDA 11.8+) [Optional but recommended]
```

### Installation

```bash
# Clone repository
git clone https://github.com/manasroy13/Automatic-object-Detection-DOTA-Dataset-.git
cd Automatic-object-Detection-DOTA-Dataset-

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── scripts/
│   ├── train.py                 # Main training script
│   ├── train_val_split.py       # Split data into train/val
│   ├── slice_images.py          # Slice large images
│   └── slice_labels_to_yolo.py  # Convert labels to YOLO format
├── .gitignore
└── README.md
```

## 🔧 Usage

### 1. Prepare Dataset

```bash
# Split DOTA into train/val
python scripts/train_val_split.py

# Convert labels to YOLO format
python scripts/slice_labels_to_yolo.py
```

### 2. Train Model

```bash
python scripts/train.py
```

**Training Parameters:**
- Model: YOLOv8s
- Image Size: 1024×1024
- Batch Size: 4
- Epochs: 100
- Device: GPU (device=0)

### 3. Run Inference

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=DOTA/test/images \
  imgsz=1024 \
  conf=0.25 \
  save=True
```

## 📊 Classes

| ID | Class | ID | Class |
|----|----|----|----|
| 0 | plane | 8 | bridge |
| 1 | ship | 9 | large-vehicle |
| 2 | storage-tank | 10 | small-vehicle |
| 3 | baseball-diamond | 11 | helicopter |
| 4 | tennis-court | 12 | roundabout |
| 5 | basketball-court | 13 | soccer-ball-field |
| 6 | ground-track-field | 14 | swimming-pool |
| 7 | harbor | | |

## 📈 Model Performance

Results will be logged in `runs/detect/train/` directory with:
- Training metrics (loss, mAP)
- Validation results
- Best model weights (`best.pt`)
- Confusion matrix
- Training plots

## 🔗 Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **DOTA Dataset**: https://captain-whu.github.io/DiRS/
- **Ultralytics**: https://github.com/ultralytics/ultralytics

## 📝 License

This project is for educational purposes. DOTA dataset is provided by Wuhan University.

## 👨‍💻 Author

Manas Roy (manasroy13)

## ⚠️ Important Notes

- Large files (models, datasets) are excluded from git (see `.gitignore`)
- Requires significant disk space (~50GB+) for full dataset
- GPU training significantly speeds up model training
- Adjust batch size based on GPU memory availability

---

**Last Updated**: February 2026
