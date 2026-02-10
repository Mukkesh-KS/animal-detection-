import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_SIZE = (96, 96)
CALIB_DIR = "calib_images"   # 50–100 real images

def representative_dataset():
    for img_name in os.listdir(CALIB_DIR)[:100]:
        img = Image.open(os.path.join(CALIB_DIR, img_name)).convert("L")
        img = img.resize(IMG_SIZE)

        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
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
