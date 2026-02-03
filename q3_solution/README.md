# Q3. Foundation of Machine Learning (4 points)

## 📋 Tasks

1. Explain underfitting and overfitting (1 point)
2. Derive logistic regression and implement from scratch (3 points)

---

## 📁 Files

```
q3_solution/
├── README.md                    # This file
├── logistic_regression.py       # From-scratch implementation
└── bias_variance_plot.png       # Visualization
```

---

## Q3.1: Underfitting & Overfitting

### Definitions
| Condition | Cause | Symptom |
|-----------|-------|---------|
| **Underfitting** | Model too simple | High bias, poor on both train/test |
| **Overfitting** | Model too complex | Low training error, high test error |

### Bias-Variance Tradeoff
- **Bias**: Error from oversimplified assumptions
- **Variance**: Sensitivity to training data fluctuations
- **Test Error** = Bias² + Variance + Irreducible Noise

---

## Q3.2: Logistic Regression

### Why "Regression"?
It performs regression on **probability** (continuous), not labels:

$$\log\frac{p(y=1|x)}{1-p(y=1|x)} = w^\top x + b$$

### Model
$$p(y=1|x) = \sigma(z) = \frac{1}{1+e^{-z}}, \quad z = w^\top x + b$$

### Loss Function (Cross-Entropy)
$$J(w,b) = -\sum_{i=1}^n \left[ y_i \log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i)) \right]$$

### Gradients
$$\frac{\partial J}{\partial w} = \sum_{i=1}^n (\sigma(z_i) - y_i)x_i$$
$$\frac{\partial J}{\partial b} = \sum_{i=1}^n (\sigma(z_i) - y_i)$$

---

## 🚀 Quick Start

```bash
python logistic_regression.py
```

### Expected Output
```
CLASSIFICATION REPORT
==================================================
Accuracy:  0.9500
Precision: 0.9474
Recall:    0.9730
F1-Score:  0.9600
```

---

## 💡 Key Insights

1. **Sigmoid** maps any real number to (0, 1)
2. **Cross-entropy** is the negative log-likelihood
3. **Gradient** has elegant form: `(prediction - label) × input`
4. **Regularization** adds $\frac{\lambda}{2}\|w\|^2$ to prevent overfitting
