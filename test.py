import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (224, 224)  # Image dimensions for the model
BATCH_SIZE = 32
MODEL_PATH = "covid_pneumonia_detection_xception.h5"  # Path to the trained model
CLASS_INDICES = {0: "COVID", 1: "Normal", 2: "Pneumonia"}  # Class mapping
TEST_DIR =  # Give Test dataset path

# Load the model
model = load_model(MODEL_PATH)

# Data generator for test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Function to predict a single X-ray image
def predict_single_image(image_path, model, class_indices):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
        predicted_class = class_indices[predicted_class_index]  # Map index to class name

        # Print the result
        print(f"The X-ray image is classified as: {predicted_class}")
    except Exception as e:
        print(f"Error processing the image: {e}")

# Path to a test image for single prediction
image_path =  # Give Test dataset path
# Predict and print the result for a single image
predict_single_image(image_path, model, CLASS_INDICES)

