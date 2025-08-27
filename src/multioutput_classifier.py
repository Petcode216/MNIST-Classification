import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def add_noise(X, noise_factor=1000):
    rnd = np.random.randint(0, noise_factor, size=X.shape)
    return X + rnd

def train_multioutput_classifier(X_train, y_train):
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    return knn_clf

def denoise_image(model, noisy_image):
    return model.predict([noisy_image])[0]
