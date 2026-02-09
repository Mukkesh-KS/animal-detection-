import cv2
import os

INPUT_DIR = r"e:\embedded\clg proj\animal-detection-\ml\code\calib_images"  # 🔴 CHANGE THIS
OUTPUT_DIR = "animal_road_96x96_gray"
IMG_SIZE = (96, 96)
MAX_IMAGES = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = os.listdir(INPUT_DIR)
print("Files found in input dir:", len(files))

count = 0
for img_name in files:
    if count >= MAX_IMAGES:
        break

    # accept common image formats
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Cannot read:", img_name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    out_path = os.path.join(OUTPUT_DIR, f"animal_{count}.jpg")
    cv2.imwrite(out_path, gray)

    count += 1

print(f"✅ Saved {count} grayscale 96x96 animal road images")
