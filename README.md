# Machine Learning based Pneumonia Classification

This project is a computer-aided diagnosis (CAD) system for the automated detection of pneumonia from chest X-ray images using machine learning techniques.

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**
    Run the following command to download the dataset and preprocess it:
    ```bash
    python src/download_dataset.py
    ```

4.  **Train the models:**
    Run the following command to train the machine learning models:
    ```bash
    python src/train_models.py
    ```

5.  **Run the application:**
    Run the following command to start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Scope

This project currently supports the classification of chest X-ray images as "Pneumonia" or "Normal". The models have been trained on a dataset that only includes these two classes. To extend the project to classify "COVID-19", a new dataset containing COVID-19 images would be required, and the models would need to be retrained on this new dataset.