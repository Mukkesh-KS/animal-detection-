from bing_image_downloader import downloader
import os

ROOT_DIR = "raw_dataset_extra"
os.makedirs(ROOT_DIR, exist_ok=True)

download_plan = {
    "cars_highway": {
        "total": 260,   # clean later to 200
        "queries": [
            "car on highway day",
            "car on highway night",
            "car driving on expressway",
            "car on highway side view"
        ]
    },

    "heavy_vehicles": {
        "total": 110,   # clean later to 80
        "queries": [
            "bus on highway",
            "truck on highway",
            "lorry on road",
            "van on highway"
        ]
    },

    "traffic_objects_extra": {
        "total": 70,    # clean later to 50
        "queries": [
            "traffic sign on highway",
            "traffic cones on road",
            "road barricade highway",
            "milestone on highway"
        ]
    },

    "static_objects_extra": {
        "total": 55,    # clean later to 40
        "queries": [
            "electric pole near road",
            "street light highway",
            "roadside fence highway",
            "parked vehicle near road"
        ]
    },

    "weather_visual_noise_extra": {
        "total": 60,    # clean later to 43
        "queries": [
            "road fog",
            "road rain",
            "road dust",
            "headlight glare night"
        ]
    }
}

print("📥 Starting balanced NO_ANIMAL image download...\n")

for category, data in download_plan.items():
    category_dir = os.path.join(ROOT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    per_query = data["total"] // len(data["queries"])

    for q in data["queries"]:
        downloader.download(
            query=q,
            limit=per_query,
            output_dir=category_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )

print("\n✅ Extra image download completed!")
