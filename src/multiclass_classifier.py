from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_sgd_classifier(X_train, y_train):
    clf = SGDClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_rf_classifier(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    return acc, cm, report

def predict_multiclass(clf, X):
    return clf.predict(X)