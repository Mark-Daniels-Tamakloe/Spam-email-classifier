# Spam Email Classifier using Structural Risk Minimization

This project implements a spam email classifier using supervised learning and structural risk minimization. The classifier is trained using multiple loss functions — Ridge Regression, Logistic Regression, and Hinge Loss — with gradient descent optimization. The model is evaluated based on its ability to distinguish spam from non-spam emails using real-world email datasets.

---

## 📌 Project Highlights

- Built and compared models using three loss functions: Ridge, Logistic, and Hinge.
- Implemented custom gradient descent with adaptive step-size.
- Achieved an AUC of **97.97%**, with a false positive rate of **0.65%** and a true positive rate of **56.09%**.
- Used time-split training/validation to simulate real-world spam detection.

---

## 🧠 Key Concepts

- **Structural Risk Minimization**: Balancing training error with model complexity.
- **Gradient Descent**: Implemented from scratch with custom step-size tuning.
- **Loss Functions**:
  - Ridge Regression (`L2` regularized squared error)
  - Logistic Regression (`log-loss`)
  - Hinge Loss (`SVM-style` margin maximization)
- **Evaluation Metrics**:
  - AUC (Area Under ROC Curve)
  - True Positive Rate (TPR)
  - False Positive Rate (FPR)

---

## 🛠 Tools & Technologies

- **Python 3.6+**
- `NumPy` for numerical computation
- Custom-built machine learning pipeline (no scikit-learn)
- ROC visualization and gradient checking utilities included

---

## 🧪 Sample Results

```text
False positive rate: 0.65%
True positive rate: 56.09%
AUC: 97.97%
