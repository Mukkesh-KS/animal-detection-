import os
import cv2
import imagehash
from PIL import Image
from ultralytics import YOLO

# ------------ CONFIG ------------
INPUT_DIR = 'animal_on_road_900'
OUTPUT_DIR = 'animal_on_road_clean'
IMG_SIZE = 96
HASH_THRESHOLD = 6   # duplicate sensitivity
# --------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 (lightweight)
model = YOLO('yolov8n.pt')

# COCO class IDs
PERSON = 0
BIRD = 14

hashes = []
saved_count = 0

def is_duplicate(pil_img):
    h = imagehash.phash(pil_img)
    for old_h in hashes:
        if abs(h - old_h) <= HASH_THRESHOLD:
            return True
    hashes.append(h)
    return False

def contains_unwanted_objects(img_path):
    results = model(img_path, conf=0.4, verbose=False)
    for r in results:
        for cls in r.boxes.cls:
            if int(cls) in [PERSON, BIRD]:
                return True
    return False

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    src_path = os.path.join(INPUT_DIR, file)

    try:
        pil_img = Image.open(src_path).convert('RGB')

        # 1️⃣ Remove duplicates
        if is_duplicate(pil_img):
            continue

        # 2️⃣ Remove humans / birds / signs (most cases)
        if contains_unwanted_objects(src_path):
            continue

        # 3️⃣ Convert to grayscale & resize
        img = cv2.imread(src_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        out_path = os.path.join(OUTPUT_DIR, f'{saved_count}.jpg')
        cv2.imwrite(out_path, resized)
        saved_count += 1

    except Exception as e:
        print(f"Skipped {file}: {e}")

print(f"✅ CLEAN DATASET READY: {saved_count} images")
