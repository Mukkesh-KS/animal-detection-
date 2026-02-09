import os
from PIL import Image

input_folder = "bus on road day"
output_folder = "bus_96python_gr"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        img_path = os.path.join(input_folder, filename)

        img = Image.open(img_path).convert("L")   # grayscale
        img = img.resize((96, 96))                # resize 96x96

        # convert name to jpg
        new_name = os.path.splitext(filename)[0] + ".jpg"
        save_path = os.path.join(output_folder, new_name)

        # JPEG needs RGB mode
        img = img.convert("RGB")
        img.save(save_path, "JPEG", quality=95)

print("✅ no_animal folder images converted to 96x96 grayscale JPEG!")
