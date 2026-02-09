# Download 200 images of "animals in road" using Bing Image Search
# Requirements:
# pip install bing-image-downloader

from bing_image_downloader import downloader

downloader.download(
    query="animals on road",
    limit=200,
    output_dir="dataset",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

print("✅ 200 images downloaded in dataset/animals on road/")
