# train.py
# STEP 3, 4, 5 – Train tiny CNN for Animal-on-Road detection (ESP32 compatible)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 15

# IMPORTANT: dataset folder path
DATASET_PATH = "dataset_final"

# -------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

# -------------------------------
# MODEL (TINY CNN – ESP32 SAFE)
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# TRAINING
# -------------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("animal_detector.h5")

print("✅ Training completed successfully")
print("✅ Model saved as animal_detector.h5")
