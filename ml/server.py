import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt

# -------------------------
# Flask + MQTT Setup
# -------------------------
app = Flask(__name__)
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)

# -------------------------
# Load Pretrained Model
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# -------------------------
# Image Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# Load ImageNet labels
# -------------------------
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# -------------------------
# LAND ANIMALS ONLY
# -------------------------
animal_keywords = [
    # --- Cattle / Buffalo / Ox ---
    "cow", "bull", "buffalo", "water buffalo",
    "ox", "musk ox", "yak","cat",

    # --- Sheep / Goat / Horned animals ---
    "goat", "sheep", "ram", "bighorn",

    # --- Horses / Camel family ---
    "horse", "camel",

    # --- Elephants ---
    "elephant",

    # --- Deer family & Antelope family ---
    "deer", "antelope", "hartebeest", "gazelle", "impala",

    # --- Dogs (ALL DOG BREEDS) ---
    "dog", "terrier", "hound", "retriever", "shepherd",
    "mastiff", "spaniel", "bulldog", "pug", "chihuahua",
    "poodle", "husky", "doberman", "rottweiler",
    "malamute", "collie",

    # --- Monkeys ---
    "monkey",

    # --- Wild animals ---
    "bear", "fox", "wolf", "jackal",
    "leopard", "tiger", "lion",

    # --- Others ---
    "pig", "boar", "hog",
    "rabbit",
    "panda",
    "mongoose",
    "hyena"
]



# ❌ SEA CREATURES (ignore)
sea_animals = [
    "fish", "shark", "whale", "dolphin", "seal", "otter",
    "crab", "lobster", "shrimp", "jellyfish", "starfish"
]

# ❌ BIRDS (ignore)
birds = [
    "bird", "eagle", "pigeon", "crow", "sparrow", "parrot",
    "owl", "duck", "hen", "rooster", "peacock"
]

# ❌ Humans
human_keywords = ["person", "human", "man", "woman", "boy", "girl"]

# -------------------------
# Classify Function
# -------------------------
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, index = outputs.max(1)
        class_id = index.item()

    class_name = labels[class_id].lower()

    # check categories
    is_land_animal = any(word in class_name for word in animal_keywords)
    is_bird = any(word in class_name for word in birds)
    is_sea = any(word in class_name for word in sea_animals)
    is_human = any(word in class_name for word in human_keywords)

    # Final decision: only land animals allowed
    if is_land_animal and not (is_bird or is_sea or is_human):
        return True, class_name
    else:
        return False, class_name

# -------------------------
# Flask Route
# -------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "uploaded.jpg"
    file.save(file_path)

    is_animal, class_name = classify_image(file_path)

    if is_animal:
        print(f"🐾 Land Animal Detected: {class_name}")
        client.publish("esp32/alert", "animal_detected")
        return jsonify({"result": "Animal", "class_name": class_name})
    else:
        print("📦 Not a land animal")
        return jsonify({"result": "Not Animal", "class_name": class_name})

# -------------------------
# Run Server
# -------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
