import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.show()

def show_misclassified_images(X, y_true, y_pred, class_names, n_images=10):
    misclassified = np.where(y_true != y_pred)[0]
    if len(misclassified) == 0:
        print("No misclassified samples found!")
        return
    idxs = np.random.choice(misclassified, min(n_images, len(misclassified)), replace=False)
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idxs, 1):
        plt.subplot(2, 5, i)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}")
        plt.axis("off")
    plt.show()
