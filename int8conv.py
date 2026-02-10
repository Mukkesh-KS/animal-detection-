import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "animal_detector_demo.h5"
CALIB_DIR = "calib_images"
IMG_SIZE = (96, 96)
MAX_CALIB_IMAGES = 300

def representative_dataset():
    count = 0
    for root, _, files in os.walk(CALIB_DIR):
        for fname in files:
            if count >= MAX_CALIB_IMAGES:
                return
            if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                continue

            img = Image.open(os.path.join(root, fname)).convert("L")
            img = img.resize(IMG_SIZE)

            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            yield [img]
            count += 1

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("animal_detector_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model saved")
