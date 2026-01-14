import tensorflow as tf
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights="imagenet")

model.save("model/leafguard_model.h5")

print("✅ Dummy model created successfully!")
