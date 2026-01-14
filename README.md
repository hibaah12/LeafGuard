## LeafGuard | AI Plant Disease Detection Web Application

LeafGuard is an AI-powered web application designed to help farmers detect plant leaf diseases quickly and accurately. The app not only identifies the disease but also provides preventive measures and fertilizer recommendations. All results can be translated into any language and announced aloud, making it highly accessible for farmers.

**Key Features**

-Accurate Disease Detection: Uses MobileNetV2-based deep learning model to classify 38 plant disease and healthy classes.

-Preventive Measures & Fertilizer Recommendations: Provides actionable guidance to protect crops and improve yield.

-Multi-Language Support: Integrated Google Translator API to translate results into any language.

-Text-to-Speech Functionality: Farmers can hear the results aloud in the selected language.

-Farmer-Friendly Interface: Simple UI with image preview for easy navigation by non-technical users.

**Technologies Used**

-Python

-TensorFlow / Keras (MobileNetV2)

-Flask

-HTML, CSS, JavaScript

-Google Translator API

-Text-to-Speech Libraries

**Installation**

1. **Clone the repository:**

```git clone https://github.com/your-username/LeafGuard.git
cd LeafGuard
```


2. **Create a virtual environment:**

- **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
- **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```


3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


 4. **Run the Flask app:**
    ```bash
    python app.py
    ```
   

 5. **Open your browser and go to:**
    ```bash
    http://127.0.0.1:5000
    ```


**Usage**

Upload a leaf image of your plant.

The app will detect the disease using MobileNetV2.

Receive fertilizer recommendations and preventive measures.

Select a language to translate the results.

Use text-to-speech to hear the results aloud.

**Project Structure**
```LeafGuard/
│
├── .venv/                 # Virtual environment (should be gitignored)
├── assets/                # Images or static assets
├── data/                  # Raw datasets
├── dataset/               # Organized dataset
├── dataset_split/         # Train/Test split datasets
├── LeafGuard/             # Main project folder
├── model/                 # MobileNetV2 trained model
├── static/                # CSS, JS, images
├── templates/             # HTML templates
├── uploads/               # Uploaded images by users
├── utils/                 # Utility scripts
├── app.py                 # Flask application
├── create_dummy_model.py  # Script to create model
├── predictor.py           # Model inference
├── split_dataset.py       # Dataset splitting script
├── train_model.py         # Model training script
└── requirements.txt       # Python dependencies
```
This project is open source and available for personal and educational use.  
Contributions are welcome! Feel free to submit issues, fork the repository, and create pull requests.  
If you run into trouble or just want to say hi, feel free to open an issue or reach out.



