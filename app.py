import os
import json
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Set working directory & model paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Google Drive model download URL (your own file ID)
gdrive_url = "https://drive.google.com/uc?id=1gXWh24HlPCVB8HWBADoRVxGBbxX-S5VI"

st.title("🌿 Plant Disease Classifier")

@st.cache_resource
def download_and_load_model():
    # Download model every time (overwrite if exists)
    st.info("Downloading model (500MB) from Google Drive, please wait...")
    gdown.download(gdrive_url, model_path, quiet=False)
    
    st.info("Loading model...")
    model = tf.keras.models.load_model(model_path)
    return model

# Load model (download + load cached in session)
model = download_and_load_model()

# Load class indices JSON once
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # if PNG with alpha channel, remove alpha
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image, class_indices):
    preprocessed = preprocess_image(image)
    preds = model.predict(preprocessed)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = class_indices[str(pred_idx)]
    return pred_label

uploaded_file = st.file_uploader("Upload a plant image (jpg/png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image.resize((200, 200)), caption="Uploaded Image", use_column_width=False)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            prediction = predict_image_class(model, image, class_indices)
            st.success(f"Prediction: {prediction}")
