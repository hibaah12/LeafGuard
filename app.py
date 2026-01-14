from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from deep_translator import GoogleTranslator
from gtts import gTTS  

app = Flask(__name__)

MODEL_PATH = "model/leafguard_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found: leafguard_model.h5")
model = load_model(MODEL_PATH)

class_labels = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Healthy Apple',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Healthy Cherry',
    'Corn Cercospora Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight',
    'Healthy Corn', 'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Healthy Grape',
    'Orange Huanglongbing', 'Peach Bacterial Spot', 'Healthy Peach',
    'Pepper Bacterial Spot', 'Healthy Pepper', 'Potato Early Blight',
    'Potato Late Blight', 'Healthy Potato', 'Raspberry Healthy',
    'Soybean Healthy', 'Squash Powdery Mildew', 'Strawberry Leaf Scorch',
    'Healthy Strawberry', 'Tomato Bacterial Spot', 'Tomato Early Blight',
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Mosaic Virus',
    'Healthy Tomato'
]

disease_info = {
    "Tomato Target Spot": {
        "fertilizer": "Use Mancozeb or Chlorothalonil weekly.",
        "prevention": "Avoid overhead watering and remove infected leaves."
    },
    "Apple Scab": {
        "fertilizer": "Use sulfur or copper fungicides.",
        "prevention": "Plant resistant varieties and prune regularly."
    },
    "Healthy Tomato": {
        "fertilizer": "Maintain balanced NPK 10-10-10 nutrients.",
        "prevention": "Consistent watering and pest monitoring."
    }
}

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file name"}), 400

        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        confidence = round(float(np.max(prediction) * 100), 2)

        info = disease_info.get(predicted_label, {
            "fertilizer": "Use balanced fertilizer regularly.",
            "prevention": "Maintain soil health and proper irrigation."
        })

        language = request.form.get("language", "en")

        text = f"The detected disease is {predicted_label}. " \
               f"Fertilizer recommendation: {info['fertilizer']}. " \
               f"Prevention: {info['prevention']}."

        translated_text = text
        if language != "en":
            translated_text = GoogleTranslator(source="auto", target=language).translate(text)

        
        speech_path = os.path.join("static", "speech.mp3")
        tts = gTTS(text=translated_text, lang=language)
        tts.save(speech_path)

        return jsonify({
            "disease": predicted_label,
            "confidence": confidence,
            "fertilizer": info["fertilizer"],
            "prevention": info["prevention"],
            "translated": translated_text,
            "audio": "/static/speech.mp3"
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)














