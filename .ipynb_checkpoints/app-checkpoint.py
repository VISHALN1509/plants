import os
import random
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib

# Load models
feature_extractor = load_model("feature_extractor_model.h5")
classifier = joblib.load("rf_model.pkl")
class_labels = np.load("class_labels.npy")

# App title
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
st.title("🌿 Plant Disease Classification App")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Feature extraction
    feature = feature_extractor.predict(img_array, verbose=0)
    prediction = classifier.predict(feature)[0]

    st.subheader("Prediction")
    st.write(f"✅ **Predicted Class:** {prediction}")

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(feature)[0]
        top_indices = np.argsort(probs)[-5:][::-1]
        st.write("Top 5 Predictions:")
        for idx in top_indices:
            st.write(f"{class_labels[idx]}: {probs[idx]*100:.2f}%")

