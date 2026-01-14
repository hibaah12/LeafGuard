import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os, shutil, glob, json

ROOT_PATH = "dataset_split"
TRAIN_PATH = os.path.join(ROOT_PATH, "train")
VAL_PATH = os.path.join(ROOT_PATH, "val")
MODEL_PATH = "model/leafguard_model.h5"

os.makedirs("model", exist_ok=True)

train_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=25,
    zoom_range=0.25,
    shear_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    VAL_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    verbose=1
)

model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

labels = list(train_gen.class_indices.keys())
with open("model/class_labels.json", "w") as f:
    json.dump(labels, f)
print("✅ Labels saved:", labels)

