import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def create_multi_labels(y):
    is_large = (y >= 7)
    is_odd = (y % 2 == 1)
    return np.c_[is_large, is_odd]

def train_multilabel_classifier(X_train, y_train):
    y_train_multi = create_multi_labels(y_train)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train_multi)
    return knn_clf

def evaluate_multilabel_classifier(model, X_test, y_test):
    y_test_multi = create_multi_labels(y_test)
    y_pred = model.predict(X_test)
    return f1_score(y_test_multi, y_pred, average="macro")

def make_multilabel_targets(y, ge_threshold=7):
    """Return 2-column labels: [is_large(>=th), is_odd]."""
    y = np.asarray(y)
    is_large = (y >= ge_threshold).astype(int)
    is_odd = (y % 2 == 1).astype(int)
    return np.c_[is_large, is_odd]

def decode_bool_label(row, ge_threshold=7):
    """Return human-readable label string for a [is_large, is_odd] row."""
    is_large = bool(row[0])
    is_odd = bool(row[1])
    large_txt = f"≥{ge_threshold}" if is_large else f"<{ge_threshold}"
    parity_txt = "odd" if is_odd else "even"
    return f"{large_txt}, {parity_txt}"
