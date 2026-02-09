from icrawler.builtin import BingImageCrawler
import os
import shutil

SAVE_DIR = 'animal_on_road_900'
ZIP_NAME = 'animal_on_road_900.zip'

os.makedirs(SAVE_DIR, exist_ok=True)

# 🔑 KEYWORD EXPANSION (this is the secret)
keywords = [
    'animal on road',
    'animal crossing road',
    'cow on road india',
    'dog on road traffic',
    'elephant on road',
    'deer crossing highway',
    'wild animal on road',
    'stray animals on road',
    'animal road accident',
]

IMAGES_PER_KEYWORD = 120   # 9 × 120 ≈ 1080 (some duplicates will be skipped)

crawler = BingImageCrawler(
    storage={'root_dir': SAVE_DIR}
)

for kw in keywords:
    print(f'🔍 Downloading: {kw}')
    crawler.crawl(
        keyword=kw,
        max_num=IMAGES_PER_KEYWORD
    )

# Count images
image_count = len(os.listdir(SAVE_DIR))
print(f"✅ Total images downloaded: {image_count}")

if image_count < 900:
    print("⚠️ Less than 900 images found, but dataset is usable.")

# ZIP
if os.path.exists(ZIP_NAME):
    os.remove(ZIP_NAME)

shutil.make_archive(
    base_name='animal_on_road_900',
    format='zip',
    root_dir='.',
    base_dir=SAVE_DIR
)

print("📦 ZIP created successfully!")
