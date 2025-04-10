import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load class names from JSON
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load the trained model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Page config
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Image Classifier")
st.write("Upload an image of an animal and the model will predict which one it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_batch)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.success(f"🐾 Predicted: **{predicted_label.capitalize()}** ({confidence:.2f}% confidence)")
