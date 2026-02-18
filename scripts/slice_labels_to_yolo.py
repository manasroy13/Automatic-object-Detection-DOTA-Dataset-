import os
import cv2
from tqdm import tqdm

SLICE_SIZE = 1024
OVERLAP = 200

CLASSES = [
    "plane","ship","storage-tank","baseball-diamond","tennis-court",
    "basketball-court","ground-track-field","harbor","bridge",
    "large-vehicle","small-vehicle","helicopter","roundabout",
    "soccer-ball-field","swimming-pool"
]

def parse_dota_label(label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            pts = list(map(float, parts[:8]))
            cls = parts[8]
            objects.append((pts, cls))
    return objects

def convert_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return xc, yc, w, h

def process_split(split):
    IMG_DIR = f"{split}/images"
    IMG_SPLIT_DIR = f"{split}/images_split"
    LABEL_DIR = f"{split}/labels"
    LABEL_SPLIT_DIR = f"{split}/labels_split"

    os.makedirs(LABEL_SPLIT_DIR, exist_ok=True)

    if not os.path.exists(IMG_SPLIT_DIR):
        print(f"Skipping {split}: {IMG_SPLIT_DIR} not found yet")
        return

    for img_name in tqdm(os.listdir(IMG_SPLIT_DIR)):
        base, idx = img_name.rsplit("_", 1)
        idx = int(idx.replace(".png", ""))

        orig_img_path = os.path.join(IMG_DIR, base + ".png")
        orig_label_path = os.path.join(LABEL_DIR, base + ".txt")
        split_img_path = os.path.join(IMG_SPLIT_DIR, img_name)

        if not os.path.exists(orig_label_path):
            continue

        img = cv2.imread(orig_img_path)
        h, w, _ = img.shape

        row = idx * (SLICE_SIZE - OVERLAP)
        col = 0
        tile_x = (idx * (SLICE_SIZE - OVERLAP)) % w
        tile_y = ((idx * (SLICE_SIZE - OVERLAP)) // w) * (SLICE_SIZE - OVERLAP)

        objs = parse_dota_label(orig_label_path)
        yolo_lines = []

        for pts, cls in objs:
            xs = pts[0::2]
            ys = pts[1::2]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            ixmin = max(xmin, tile_x)
            iymin = max(ymin, tile_y)
            ixmax = min(xmax, tile_x + SLICE_SIZE)
            iymax = min(ymax, tile_y + SLICE_SIZE)

            if ixmax <= ixmin or iymax <= iymin:
                continue

            ixmin -= tile_x
            ixmax -= tile_x
            iymin -= tile_y
            iymax -= tile_y

            cls_id = CLASSES.index(cls)
            xc, yc, bw, bh = convert_to_yolo(
                ixmin, iymin, ixmax, iymax, SLICE_SIZE, SLICE_SIZE
            )

            if bw > 0 and bh > 0:
                yolo_lines.append(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

        with open(os.path.join(LABEL_SPLIT_DIR, img_name.replace(".png", ".txt")), "w") as f:
            f.write("\n".join(yolo_lines))


# Process TRAIN and VAL
process_split("train")
process_split("val")
