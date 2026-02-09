import os, zipfile, hashlib
from PIL import Image
from bing_image_downloader import downloader

# ✅ Category -> (target count, search queries)
categories = {
    "humans_pedestrians": (200, [
        "person walking near road day",
        "person walking near road night",
        "pedestrians crossing road day",
        "crowd near roadside day",
        "roadside worker near road day",
        "roadside worker near road night"
    ]),

    "road_infrastructure": (250, [
        "empty road day",
        "empty road night",
        "road divider day",
        "toll gate day",
        "toll plaza night",
        "flyover road day",
        "bridge road night"
    ]),

    "traffic_objects": (150, [
        "traffic signboard on road day",
        "traffic cones on road",
        "road barricade day",
        "road barricade night",
        "milestone on highway"
    ]),

    "weather_visual_noise": (200, [
        "road fog day",
        "road fog night",
        "road rain day",
        "road rain night",
        "road dust smoke day",
        "road glare night",
        "road shadows day"
    ]),

    "static_objects": (100, [
        "electric poles near road day",
        "street lights on road night",
        "fence near highway day",
        "parked vehicle near road day",
        "parked vehicle near road night"
    ])
}

RAW_ROOT = "raw_dataset"
FINAL_ROOT = "final_dataset_900"
ZIP_NAME = "dataset_900.zip"

os.makedirs(RAW_ROOT, exist_ok=True)
os.makedirs(FINAL_ROOT, exist_ok=True)

# ✅ Step 1: Download images
print("📥 Downloading images from Bing...")
for cat, (target, queries) in categories.items():
    for q in queries:
        downloader.download(
            query=q,
            limit=70,  # download extra to handle duplicates/bad images
            output_dir=os.path.join(RAW_ROOT, cat),
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )

print("✅ Download complete!")

# ✅ Step 2: Remove duplicates + Convert images for each category
print("🧹 Cleaning duplicates + converting to 96x96 grayscale JPEG...")

seen_hashes = set()  # ✅ global duplicates removed across all categories too

for cat, (target, _) in categories.items():
    input_root = os.path.join(RAW_ROOT, cat)
    output_cat_folder = os.path.join(FINAL_ROOT, cat)
    os.makedirs(output_cat_folder, exist_ok=True)

    # collect all images in category subfolders
    all_images = []
    for root, dirs, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                all_images.append(os.path.join(root, f))

    count = 0
    for img_path in all_images:
        if count >= target:
            break

        try:
            img = Image.open(img_path).convert("RGB")

            # ✅ duplicate check (exact)
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            # ✅ grayscale + resize + save JPEG
            img = img.convert("L").resize((96, 96)).convert("RGB")

            save_path = os.path.join(output_cat_folder, f"{cat}_{count+1:04d}.jpg")
            img.save(save_path, "JPEG", quality=95)

            count += 1

        except:
            continue

    print(f"✅ {cat}: saved {count}/{target}")

print("✅ All categories processed!")

# ✅ Step 3: Zip final dataset
print("📦 Creating ZIP...")
with zipfile.ZipFile(ZIP_NAME, "w") as zipf:
    for root, dirs, files in os.walk(FINAL_ROOT):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, FINAL_ROOT)
            zipf.write(full_path, arcname=arcname)

print(f"✅ ZIP created: {ZIP_NAME}")
print("🎉 Done!")
