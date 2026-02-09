import os
import hashlib
import zipfile
from PIL import Image

# ================= CONFIG =================
INPUT_ROOT = "dataset_non_animal"   # images directly inside this folder
OUTPUT_ROOT = "noanimal_dataset_calib"
ZIP_NAME = "noanimal_dataset_calib.zip"

TARGET_COUNT = 1500
IMG_SIZE = (96, 96)
# ==========================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)
output_class = os.path.join(OUTPUT_ROOT, "animal")
os.makedirs(output_class, exist_ok=True)

seen_hashes = set()
saved_count = 0

print("📂 INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))
print("🖼️ Processing images...")

# ================= PROCESS =================
for root, _, files in os.walk(INPUT_ROOT):
    for file in files:
        if saved_count >= TARGET_COUNT:
            break

        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        img_path = os.path.join(root, file)

        try:
            # Load image
            img = Image.open(img_path).convert("RGB")

            # Duplicate detection (before resize)
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            # Convert to grayscale + resize
            img = img.convert("L").resize(IMG_SIZE).convert("RGB")

            saved_count += 1
            save_path = os.path.join(
                output_class, f"animal_{saved_count:04d}.jpg"
            )

            img.save(save_path, "JPEG", quality=95)

        except Exception as e:
            print("❌ Skipped:", img_path, e)

    if saved_count >= TARGET_COUNT:
        break

print(f"\n🎉 FINAL TOTAL images saved: {saved_count} (EXPECTED: {TARGET_COUNT})")

# ================= ZIP DATASET =================
print("\n📦 Creating ZIP...")
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_ROOT):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, OUTPUT_ROOT)
            zipf.write(full_path, arcname)

print(f"✅ ZIP created: {ZIP_NAME}")
