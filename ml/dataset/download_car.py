from bing_image_downloader import downloader

downloader.download(
    query="car on highway at night all angles",
    limit=120,              # download extra to remove duplicates
    output_dir="dataset",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

print("✅ Download complete!")
