import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops

def extract_hog_features(image):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.
    """
    # Resize image for consistency
    image = cv2.resize(image, (128, 128))
    # Convert to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
    return features

def extract_glcm_features(image):
    """
    Extracts Gray-Level Co-occurrence Matrix (GLCM) features from an image.
    """
    # Resize image for consistency
    image = cv2.resize(image, (128, 128))
    # Convert to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_features_from_image(image):
    """
    Extracts both HOG and GLCM features from an image object and concatenates them.
    """
    hog_features = extract_hog_features(image)
    glcm_features = extract_glcm_features(image)

    return np.concatenate((hog_features, glcm_features))
