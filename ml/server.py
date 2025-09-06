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
client.connect("broker.hivemq.com", 1883, 60)  # free public MQTT broker

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
# Keywords for animals (exclude humans)
# -------------------------
animal_keywords = [
    "dog", "cat", "bird", "cow", "sheep", "horse", "elephant", "zebra",
    "giraffe", "lion", "tiger", "bear", "monkey", "panda", "wolf", "fox",
    "rabbit", "deer", "pig", "goat", "leopard", "cheetah", "camel", "kangaroo",
    "insect", "spider", "snake", "fish", "whale", "dolphin", "otter", "seal",
    "crab", "lobster", "butterfly", "bee", "ant", "mouse", "rat", "bat","snake"
]

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

    # detect animals only (exclude humans)
    is_animal = any(word in class_name for word in animal_keywords)
    is_human = any(word in class_name for word in human_keywords)

    return is_animal and not is_human, class_name

# -------------------------
# Flask route
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
        print(f"🐾 Animal detected: {class_name}")
        client.publish("esp32/alert", "animal_detected")
        return jsonify({"result": "Animal", "class_name": class_name})
    else:
        print("📦 Not an animal")
        return jsonify({"result": "Not Animal"})

# -------------------------
# Run server
# -------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)