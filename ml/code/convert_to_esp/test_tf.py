import numpy as np
import tensorflow as tf
from PIL import Image

# ================= CONFIG =================
MODEL_PATH = "animal_detector_int8.tflite"
IMAGE_PATH = "t3.jpg"
IMG_SIZE = (96, 96)
THRESHOLD = 0.5
# ==========================================

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== MODEL INFO ===")
print("Input:", input_details)
print("Output:", output_details)

# ---------------- IMAGE PREPROCESS ----------------
img = Image.open(IMAGE_PATH).convert("L")   # grayscale
img = img.resize(IMG_SIZE)

img = np.array(img).astype(np.float32)
img = img / 255.0                           # normalize

# Quantization params
in_scale, in_zero = input_details[0]["quantization"]

# Apply INT8 quantization
img = img / in_scale + in_zero
img = np.clip(img, -128, 127).astype(np.int8)

# Shape: (1, 96, 96, 1)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)

# ---------------- INFERENCE ----------------
interpreter.set_tensor(input_details[0]["index"], img)
interpreter.invoke()

raw_output = interpreter.get_tensor(output_details[0]["index"])

# Output quant params
out_scale, out_zero = output_details[0]["quantization"]

# Dequantize output
confidence = (raw_output.astype(np.float32) - out_zero) * out_scale

print("\n=== OUTPUT DEBUG ===")
print("Raw INT8 output:", raw_output)
print("Output scale:", out_scale)
print("Output zero:", out_zero)
print("Dequantized confidence:", confidence)

confidence = confidence[0][0]

# ---------------- RESULT ----------------
print("\n=== RESULT ===")
if confidence <= THRESHOLD:
    print(f"✅ ANIMAL DETECTED ({confidence:.3f})")
else:
    print(f"❌ NOT ANIMAL ({confidence:.3f})")
