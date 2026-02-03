# Q2. Basic Statistics (2 points)

## 📋 Task

Explain the meaning of p-value and why multiple testing corrections are needed.

---

## 📁 Files

```
q2_solution/
└── README.md  # This file (theory-based question)
```

---

## 📊 What is a P-value?

A p-value is the probability of observing data as extreme as what was observed, **assuming the null hypothesis is true**:

$$p = \mathbb{P}(\text{data as extreme as observed} \mid H_0)$$

### Interpretation
| P-value | Interpretation |
|---------|----------------|
| ≤ 0.05 | Evidence against $H_0$ |
| > 0.05 | Not enough evidence to reject $H_0$ |

> **Important**: A p-value is NOT the probability that $H_0$ is true!

---

## ⚠️ Multiple Testing Problem

Running $m$ independent tests at $\alpha = 0.05$:

$$\mathbb{P}(\text{≥1 false positive}) = 1 - (1-\alpha)^m$$

For $m = 100$ tests: $1 - (0.95)^{100} \approx 0.994$

**Almost guaranteed to get a false positive!**

---

## 🛠 Correction Methods

### 1. Bonferroni (FWER control)
- Use $\alpha' = \alpha/m$
- Very conservative

### 2. Holm-Bonferroni (FWER, less conservative)
- Sort p-values, apply sequential thresholds
- Reject until $p_{(k)} > \alpha/(m-k+1)$

### 3. Benjamini-Hochberg (FDR control)
- Controls **expected proportion** of false positives among rejections
- Find largest $k$ where $p_{(k)} \leq \frac{k}{m}q$
- Recommended for discovery-oriented research
