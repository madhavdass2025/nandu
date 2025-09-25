import streamlit as st
import cv2
import numpy as np
import joblib
from src.feature_extraction import extract_features_from_image
from PIL import Image

# Load the trained models
rf_model = joblib.load('models/random_forest_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')

st.title("Pneumonia and COVID-19 Classification from Chest X-ray Images")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(opencv_image, channels="BGR", caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Extract features
    # Extract features
    features = extract_features_from_image(opencv_image)
    features = features.reshape(1, -1)

    # Make predictions
    rf_prediction = rf_model.predict(features)
    svm_prediction = svm_model.predict(features)

    st.write(f"Random Forest Prediction: **{rf_prediction[0]}**")
    st.write(f"SVM Prediction: **{svm_prediction[0]}**")