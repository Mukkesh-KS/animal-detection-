from icrawler.builtin import DuckduckgoImageCrawler
import os

# ========== CONFIG ==========
OUTPUT_DIR = "raw_animal_road_images"
NUM_IMAGES = 150
KEYWORDS = [
    "animal on road",
    "cow on road traffic",
    "dog crossing road",
    "goat on road street",
    "deer on highway road",
    "elephant crossing road"
]
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

crawler = DuckduckgoImageCrawler(
    storage={"root_dir": OUTPUT_DIR}
)

per_keyword = NUM_IMAGES // len(KEYWORDS)

for keyword in KEYWORDS:
    print(f"🔍 Downloading: {keyword}")
    crawler.crawl(
        keyword=keyword,
        max_num=per_keyword
    )

print("✅ Download completed")
