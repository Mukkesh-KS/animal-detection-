import numpy as np
import tensorflow as tf
from PIL import Image

# ================= CONFIG =================
MODEL_PATH = "animal_detector_int8.tflite"
IMAGE_PATH = "fft4.jpg"
IMG_SIZE = (96, 96)
THRESHOLD = 0.44  # animal probability threshold
# =========================================

# -------- Load TFLite Interpreter --------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== MODEL INFO ===")
print("Input dtype :", input_details[0]["dtype"])
print("Input quant :", input_details[0]["quantization"])
print("Output quant:", output_details[0]["quantization"])

# -------- Image Preprocessing (CORRECT) --------
img = Image.open(IMAGE_PATH).convert("L").resize(IMG_SIZE)
img = np.array(img, dtype=np.uint8)

in_scale, in_zero = input_details[0]["quantization"]

# Match training preprocessing (float model used /255.0)
img = img.astype(np.float32) / 255.0

# Quantize float → int8
img = img / in_scale + in_zero
img = np.round(img)
img = np.clip(img, -128, 127).astype(np.int8)

# Shape: (1, 96, 96, 1)
img = img[None, ..., None]

# -------- Inference --------
interpreter.set_tensor(input_details[0]["index"], img)
interpreter.invoke()

raw_output = interpreter.get_tensor(output_details[0]["index"])

# -------- Dequantize Output --------
out_scale, out_zero = output_details[0]["quantization"]
confidence = (raw_output.astype(np.float32) - out_zero) * out_scale
confidence = confidence[0][0]

print("\n=== OUTPUT DEBUG ===")
print("Raw INT8 output :", raw_output)
print("Dequantized conf:", confidence)

# -------- Result (CORRECT LOGIC) --------
print("\n=== RESULT ===")
if confidence <= THRESHOLD:
    print(f"✅ ANIMAL DETECTED ({confidence:.3f})")
else:
    print(f"❌ NOT ANIMAL ({confidence:.3f})")
