import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_mnist_data(test_size=10000, random_state=42):
    """Load MNIST dataset and split into train/test sets."""
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(np.uint8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def get_data_augmenter():
    """Return an ImageDataGenerator for MNIST augmentation."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )

def clean_data(X_train, X_test):
    """Check for missing values and normalize pixel values."""
    assert not np.isnan(X_train).any(), "NaNs in training set"
    assert not np.isnan(X_test).any(), "NaNs in test set"

    # Normalize to [0,1]
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    return X_train_norm, X_test_norm
