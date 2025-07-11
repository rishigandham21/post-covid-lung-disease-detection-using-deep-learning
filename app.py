import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "lung_covid_pneumonia_detection_xception.keras"
model = load_model(MODEL_PATH)

# Class labels
class_labels = ["COVID-19", "Normal", "Pneumonia"]

# Precaution recommendations
precautions = {
    "COVID-19": [
        "Isolate and monitor symptoms regularly.",
        "Stay hydrated and take prescribed medication.",
        "Use a pulse oximeter and seek help if oxygen drops."
    ],
    "Normal": [
        "Maintain a healthy diet and routine exercise.",
        "Avoid smoking and exposure to pollution.",
        "Get regular health checkups."
    ],
    "Pneumonia": [
        "Take complete rest and prescribed antibiotics.",
        "Use a humidifier to ease breathing.",
        "Drink warm fluids and avoid cold exposure."
    ]
}

# Function to preprocess input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict X-ray image
def predict_xray(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    label = class_labels[predicted_class]
    return label, precautions[label]

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        
        # Save the uploaded file to the UPLOAD_FOLDER
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Predict class and get precautions
        label, precaution_list = predict_xray(file_path)

        # Pass the predictions and file path to the template
        return render_template("upload.html", label=label, precautions=precaution_list, file_path="uploads/" + file.filename)

    return render_template("upload.html", label=None, precautions=None, file_path=None)

if __name__ == "__main__":
    app.run(debug=True)
