import os
import shutil
import random
from pathlib import Path

random.seed(42)

# Create temporary split for training
train_src = Path("e:/Dota_yolo_project/dataset/images/train")
val_dst = Path("e:/Dota_yolo_project/dataset/images/val")
labels_train_src = Path("e:/Dota_yolo_project/dataset/labels/train")
labels_val_dst = Path("e:/Dota_yolo_project/dataset/labels/val")

# Get all training images
all_images = [f for f in os.listdir(train_src) if f.endswith('.png')]
random.shuffle(all_images)

# 80/20 split
split_idx = int(len(all_images) * 0.8)
val_images = all_images[split_idx:]

# Move validation files
for img in val_images:
    src_img = train_src / img
    dst_img = val_dst / img
    shutil.move(str(src_img), str(dst_img))
    
    # Move corresponding label
    label_name = img.replace('.png', '.txt')
    src_label = labels_train_src / label_name
    dst_label = labels_val_dst / label_name
    
    if src_label.exists():
        shutil.move(str(src_label), str(dst_label))

print(f"Split complete: {len(all_images) - len(val_images)} train, {len(val_images)} val")
