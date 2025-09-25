import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from src.feature_extraction import extract_features_from_image

def load_data(data_dir):
    features = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                # Ensure it's a file before processing
                if os.path.isfile(image_path):
                    # Extract features and append to the list
                    try:
                        image = cv2.imread(image_path)
                        img_features = extract_features_from_image(image)
                        features.append(img_features)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    return np.array(features), np.array(labels)

def train_and_save_models():
    # Load training data
    train_features, train_labels = load_data('data/train')

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_features, train_labels)
    joblib.dump(rf_classifier, 'models/random_forest_model.pkl')
    print("Random Forest model trained and saved.")

    # Train SVM Classifier
    svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
    svm_classifier.fit(train_features, train_labels)
    joblib.dump(svm_classifier, 'models/svm_model.pkl')
    print("SVM model trained and saved.")

if __name__ == "__main__":
    train_and_save_models()