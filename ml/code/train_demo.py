import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 96
THRESHOLD = 0.45   # demo-optimized

model = tf.keras.models.load_model("animal_detector_demo.h5")

# Load image
img = cv2.imread("test.jpg")   # <-- your deer image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype("float32") / 255.0

# Shape: (1, 96, 96, 1)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)

prediction = model.predict(img)[0][0]

print("Raw prediction:", prediction)

if prediction < THRESHOLD:
    print("🟢 ANIMAL DETECTED")
else:
    print("🔴 NO ANIMAL")
