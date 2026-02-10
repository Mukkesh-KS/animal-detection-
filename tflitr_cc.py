# tflite_to_cc.py
import sys

INPUT_FILE = "animal_detector_int8.tflite"
OUTPUT_FILE = "animal_detector_model.cc"

with open(INPUT_FILE, "rb") as f:
    data = f.read()

with open(OUTPUT_FILE, "w") as f:
    f.write("const unsigned char animal_detector_int8_tflite[] = {\n")
    for i, b in enumerate(data):
        f.write(f"0x{b:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(
        f"const unsigned int animal_detector_int8_tflite_len = {len(data)};\n"
    )

print("✅ Converted to animal_detector_model.cc")
