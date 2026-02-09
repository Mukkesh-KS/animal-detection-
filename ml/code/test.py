import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = 96
MODEL_PATH = "animal_detector_demo.h5"
IMAGE_PATH = "sample_96x96_gray.jpg"

THRESHOLD = 0.60   # demo-optimized

model = tf.keras.models.load_model(MODEL_PATH)

img = image.load_img(
    IMAGE_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale"
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
confidence = prediction[0][0]

print("Raw output:", confidence)

if confidence < THRESHOLD:
    print("🟢 ANIMAL detected on road")
else:
    print("🔴 NO animal detected")
