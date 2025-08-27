from sklearn.linear_model import SGDClassifier

def train_binary_classifier(X_train, y_train, max_iter=1000, random_state=42):
    """
    Train a binary classifier to detect digit '5' vs 'not 5'.
    """
    # Convert labels to binary
    y_train_5 = (y_train == 5)

    clf = SGDClassifier(random_state=random_state, max_iter=max_iter)
    clf.fit(X_train, y_train_5)
    return clf

def predict_binary(clf, X):
    """Predict binary labels for '5' vs 'not 5'"""
    return clf.predict(X)
