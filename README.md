# MNIST Capstone Project 🧠🔢

This project explores **digit classification with the MNIST dataset**.  
While MNIST is a classic dataset, we increase its complexity step by step, making it a **capstone-level project** by covering:

- Binary classification (e.g., "5" vs. "not 5")
- Performance evaluation (cross-validation, confusion matrix, precision/recall, ROC)
- Multi-class classification (digits 0–9)
- Error analysis
- Multi-label classification (e.g., even/odd AND ≥5)
- Multi-output classification (image denoising task)
- Data augmentation for real-world complexity

---

## 📂 Project Structure
```bash
  MNIST_Classification/
  │
  ├── data/ # Dataset (raw + processed)
  ├── notebooks/ # Step-by-step Jupyter notebooks
  ├── src/ # Reusable Python modules
  ├── results/ # Plots, metrics, and reports
  ├── requirements.txt # Python dependencies
  └── README.md # Documentation
```

---

## 🚀 Installation

Clone this repo:
```bash
git clone https://github.com/your-username/MNIST-Capstone-Project.git
cd MNIST-Capstone-Project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Workflow

Data Ingesting & Wrangling → Load and preprocess MNIST.

Augmentation → Apply transformations (rotation, shift, zoom).

Binary Classification → Train an SGDClassifier to detect digit "5".

Performance Evaluation → Use accuracy, confusion matrix, precision, recall, ROC.

Multi-class Classification → Classify all digits 0–9.

Error Analysis → Identify misclassified digits.

Multi-label Classification → Predict multiple properties per digit.

Multi-output Classification → Perform image denoising with KNN.

---

## 🔮 Extensions

Implement CNNs (Keras/PyTorch) for higher accuracy.

Try transfer learning with pre-trained models.

Deploy as a simple Flask/FastAPI app for digit recognition.

---

## ⚙️ Requirements
```bash
pip install -r requirements.txt
```
