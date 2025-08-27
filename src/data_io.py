import os
import numpy as np

# Get project root (parent of current file's directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

def save_raw_data(X_train, y_train, X_test, y_test):
    np.save(os.path.join(RAW_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(RAW_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RAW_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(RAW_DIR, "y_test.npy"), y_test)

def load_raw_data():
    X_train = np.load(os.path.join(RAW_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(RAW_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))
    return X_train, y_train, X_test, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    np.save(os.path.join(PROC_DIR, "X_train_processed.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "X_test_processed.npy"), X_test)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROC_DIR, "y_test.npy"), y_test)

def load_processed_data():
    X_train = np.load(os.path.join(PROC_DIR, "X_train_processed.npy"))
    X_test = np.load(os.path.join(PROC_DIR, "X_test_processed.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test
