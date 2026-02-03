# Q1. Foundation of Data Science (2 points)

## 📋 Task

Write a script to randomly generate 1000 points on the surface of a sphere in 3D and 100D space. Create histograms of all pairwise distances. Explain the "Curse of Dimensionality" and its effects on ML.

---

## 📁 Files

```
q1_solution/
├── README.md                    # This file
└── curse_of_dimensionality.py  # Main script
```

---

## 🚀 Quick Start

```bash
pip install numpy matplotlib scipy
python curse_of_dimensionality.py
```

**Output**: `curse_of_dimensionality.png`

---

## 📊 Key Observations

### Distance Distribution
- **3D**: Wide distribution of distances (0 to ~2)
- **100D**: Sharply concentrated around √2 ≈ 1.414

### Mathematical Explanation
For points $x, y$ on unit sphere $S^{d-1}$:

$$\|x-y\|^2 = 2 - 2\cos\theta \approx 2 \text{ as } d \to \infty$$

Because random vectors become nearly orthogonal ($x^\top y \to 0$) in high dimensions.

---

## 💡 What is the Curse of Dimensionality?

1. **Data Sparsity**: Points become sparse in high-dimensional space
2. **Distance Concentration**: All distances become similar
3. **Overfitting Risk**: More features = more degrees of freedom to fit noise

---

## 🛠 Mitigation Strategies

1. **Dimensionality Reduction**: PCA, autoencoders
2. **Feature Selection**: Remove irrelevant features
3. **Regularization**: L1/L2 penalties
4. **More Data**: Exponentially more samples needed
5. **Domain Knowledge**: Choose features carefully
