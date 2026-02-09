from bing_image_downloader import downloader

QUERIES = [
    "road with cars",
    "highway traffic cars",
    "van on road traffic",
    "truck on highway",
    "lorry on road traffic",
    "busy city road vehicles",
    "empty road highway"
]

IMAGES_PER_QUERY = 30   # 7 × 30 ≈ 210 images
OUTPUT_DIR = "dataset_non_animal"

for query in QUERIES:
    print(f"⬇️ Downloading: {query}")
    downloader.download(
        query=query,
        limit=IMAGES_PER_QUERY,
        output_dir=OUTPUT_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("✅ Non-animal road images download complete")
