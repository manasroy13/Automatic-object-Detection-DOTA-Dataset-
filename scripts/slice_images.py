import cv2
import os
from tqdm import tqdm

SLICE_SIZE = 1024
OVERLAP = 200

def slice_images(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(src_dir)):
        img_path = os.path.join(src_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape
        count = 0

        for y in range(0, h, SLICE_SIZE - OVERLAP):
            for x in range(0, w, SLICE_SIZE - OVERLAP):
                tile = img[y:y+SLICE_SIZE, x:x+SLICE_SIZE]

                if tile.shape[0] < SLICE_SIZE or tile.shape[1] < SLICE_SIZE:
                    continue

                tile_name = f"{img_name[:-4]}_{count}.png"
                cv2.imwrite(os.path.join(dst_dir, tile_name), tile)
                count += 1


# Slice TRAIN images
slice_images("train/images", "train/images_split")

# Slice VAL images
slice_images("val/images", "val/images_split")
