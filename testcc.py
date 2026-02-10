import numpy as np
import tensorflow as tf
from PIL import Image

# ===== CONFIG =====
MODEL_CC = "animal_detector_model.cc"   # your .cc file with uint8_t model[]
IMAGE_PATH = "t4.jpg"
IMG_SIZE = (96, 96)
THRESHOLD = 0.44  # 0=animal, 1=not animal

# ===== STEP 1: Load your .cc as bytes =====
# Convert your C array to Python bytes manually or via preprocessing script
# Example: model_data = np.array([...], dtype=np.uint8).tobytes()
# If you still have the original .tflite, just use that:
MODEL_TFLITE = "animal_detector_int8.tflite"

# ===== STEP 2: Load TFLite interpreter =====
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# ===== STEP 3: Load & preprocess image =====
img = Image.open(IMAGE_PATH).convert("L").resize(IMG_SIZE)
img = np.array(img, dtype=np.float32) / 255.0  # normalize

# Quantization params
in_scale, in_zero = input_details[0]['quantization']

# Quantize float -> int8
img = img / in_scale + in_zero
img = np.clip(np.round(img), -128, 127).astype(np.int8)

# Shape: (1, 96, 96, 1)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)

# ===== STEP 4: Run inference =====
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

raw_output = interpreter.get_tensor(output_details[0]['index'])

# Dequantize
out_scale, out_zero = output_details[0]['quantization']
confidence = (raw_output.astype(np.float32) - out_zero) * out_scale
confidence = confidence[0][0]

print("Raw INT8 output:", raw_output)
print("Dequantized confidence:", confidence)

# ===== STEP 5: Apply your logic =====
if confidence <= THRESHOLD:
    print("✅ ANIMAL DETECTED")
else:
    print("❌ NOT ANIMAL")
