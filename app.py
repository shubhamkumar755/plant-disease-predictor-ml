import os
import json
import time
import gdown
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Setup working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# Google Drive model download URL
gdrive_url = "https://drive.google.com/uc?id=1gXWh24HlPCVB8HWBADoRVxGBbxX-S5VI"

# Download the model if not already downloaded
if not os.path.exists(model_path):
    st.warning("Model not found. Downloading model (500MB)... ")
    start_time = time.time()

    # Show progress bar
    with st.spinner("Downloading model... This may take upto 5min depending on your network."):
        gdown.download(gdrive_url, model_path, quiet=False)

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    st.success(f"✅ Model downloaded in {duration} seconds.")

# Load the pretrained model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices = json.load(open(class_indices_path))

# Image preprocessing function
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.
    return img_array

# Prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit UI
st.title("🌿 Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload a plant image (jpg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {prediction}")
