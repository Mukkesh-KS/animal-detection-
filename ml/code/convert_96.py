from PIL import Image

# Open image and convert to grayscale
img = Image.open("car4.jpg").convert("L")

# Resize to 96x96 (tuple is IMPORTANT)
img = img.resize((96, 96))

# Save converted image
img.save("sample_96x96_gray.jpg")

print("✅ Image converted to 96x96 grayscale")
