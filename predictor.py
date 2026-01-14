import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PlantDiseasePredictor:
    def __init__(self, model_path, labels_path="model/class_labels.json"):
        # Load trained model and labels
        self.model = load_model(model_path)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

    def predict(self, img_path):
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Make prediction
        predictions = self.model.predict(img_array)
        class_index = np.argmax(predictions)
        predicted_label = self.labels[class_index]
        return predicted_label



