from bing_image_downloader import downloader

queries = [
    "bus on road day",
    "bus on road night",
    "truck on highway day",
    "truck on highway night",
    "lorry on road day",
    "lorry on road night",
    "motorcycle on road day",
    "motorcycle on road night",
    "auto rickshaw on road day",
    "auto rickshaw on road night",
    "tractor on road day",
    "tractor on road night"
]

for q in queries:
    downloader.download(
        query=q,
        limit=80,                 # download extra (we will filter later)
        output_dir="raw_dataset",
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("✅ Download completed!")
