# Q5. Data Analysis and Machine Learning Model Construction (7 points)

## 📋 Overview

Complete machine learning pipeline for **cfDNA (cell-free DNA) fragmentomics classification** to detect cancer types from blood samples. The pipeline includes exploratory data analysis (EDA), feature engineering, model training with cross-validation, and ensemble methods.

---

## 📁 File Structure

```
q5_solution/
├── README.md                   # This file
├── src/                        # Source code
│   ├── eda.py                 # Exploratory data analysis
│   ├── stage0_data_load.py    # Data loading and preparation
│   ├── stage1_preprocessing.py # Quality control and preprocessing
│   ├── stage2_feature_selection.py # Feature selection pipeline
│   ├── voting_classifier.py   # Voting ensemble model
│   ├── specialist_ensemble.py # Specialist classifiers for hard classes
│   ├── repeated_kfold_cv.py   # Robustness validation
│   └── compute_auc_accuracy.py # Metrics computation
├── data/                      # Processed data files
├── results/                   # Model outputs and figures
├── out/                       # EDA outputs (PCA plots, stats)
├── DATA_DISCOVERY.md          # Detailed data analysis notes
├── PROJECT_REPORT.md          # Technical report
└── FINAL_REPORT.md           # Summary of findings
```

---

## 📊 Dataset Description

**Source**: cfDNA fragmentomics data from cancer patients and healthy controls.

### Class Distribution
| Class | Samples | Percentage |
|-------|---------|------------|
| Control (Healthy) | 70 | 23.3% |
| Breast Cancer | 50 | 16.7% |
| CRC (Colorectal) | 50 | 16.7% |
| Gastric | 50 | 16.7% |
| Liver | 30 | 10.0% |
| Lung | 50 | 16.7% |
| **Total** | **300** | **100%** |

### Feature Groups
| Group | Description | Features |
|-------|-------------|----------|
| EM | End Motif patterns (4-mer DNA sequences) | 256 |
| FLEN | Fragment Length distribution (50-350 bp) | 301 |
| NUCLEOSOME | Nucleosome positioning around TSS | 601 |
| **Total** | | **1,158** |

### Key Challenges
- **High dimensionality**: 1,158 features vs 300 samples (ratio 3.86:1)
- **Class imbalance**: Liver class has only 30 samples
- **Feature redundancy**: NUCLEOSOME features are highly correlated

---

## 🔄 Pipeline Architecture

```
Stage 0: Data Preparation
    ↓
Stage 1: Quality Control (on train data only)
    • Zero-variance filter → -9 features
    • Correlation filter (r>0.90) → -608 features
    • StandardScaler (fit on TRAIN only)
    └── Result: 1,158 → 541 features
    ↓
Stage 2: Feature Selection
    • Layer A - PLS-DA VIP: 541 → 120 features
    • Layer B - Stability Selection (LASSO): 120 → 38 features
    • Layer C - Group-aware PCA: 38 → 15 features
    └── Total reduction: 98.7%
    ↓
Stage 3: Model Training
    • Stratified 5-fold cross-validation
    • Hyperparameter tuning (GridSearchCV)
    • Ensemble methods (soft voting)
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Run the Pipeline
```bash
# 1. EDA
python src/eda.py

# 2. Full pipeline with voting classifier
python src/voting_classifier.py

# 3. Specialist ensemble for improved Gastric detection
python src/specialist_ensemble.py

# 4. Robustness validation
python src/repeated_kfold_cv.py
```

---

## 📈 Results Summary

### Baseline Models (5-Fold CV, before tuning)
| Model | F1 Macro | Accuracy |
|-------|----------|----------|
| **SVM (RBF)** | **0.439 ± 0.049** | 0.442 |
| Logistic Regression | 0.422 ± 0.045 | 0.425 |
| Random Forest | 0.421 ± 0.053 | 0.425 |
| XGBoost | 0.421 ± 0.050 | 0.421 |

### After Hyperparameter Tuning
| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| Logistic Regression | 0.422 | **0.455** | +0.033 |
| SVM (RBF) | 0.439 | **0.469** | +0.030 |
| SVM (Linear) | 0.396 | 0.438 | +0.042 |

### Final Model: Voting + Specialists
| Metric | Value | 95% CI |
|--------|-------|--------|
| **F1 Macro** | **0.475 ± 0.044** | [0.437, 0.513] |
| Accuracy | 0.471 ± 0.047 | [0.430, 0.512] |
| **AUC (macro, OvR)** | **0.794 ± 0.026** | [0.771, 0.817] |

### Per-Class F1 Scores
| Class | Voting Only | Voting + Specialists |
|-------|-------------|---------------------|
| Control | 0.512 | 0.510 |
| Breast | 0.482 | 0.480 |
| CRC | 0.456 | 0.458 |
| **Gastric** | 0.338 | **0.378** (+12%) |
| Liver | 0.640 | 0.638 |
| Lung | 0.421 | 0.419 |

---

## 🔬 Key Findings

1. **Feature Reduction**: Reduced features by 98.7% (1,158 → 15) while maintaining performance
2. **Gastric Detection**: Improved by 12% using specialist classifiers
3. **Model Stability**: Standard deviation reduced by 32% with specialist ensemble
4. **Best Performer**: Liver class achieves highest F1 (0.64) despite smallest sample size
5. **Label Noise**: ~2.9% of samples (7/300) identified as potentially mislabeled

---

## 📊 Confusion Matrix Insights

```
              Predicted
           Ctr  Bre  CRC  Gas  Liv  Lun
True Ctr    31    7    8    4    1    5
     Bre     8   20    5    4    1    2
     CRC     8    4   18    3    1    6
     Gas     7    8    4   12    4    5
     Liv     2    1    0    3   16    2
     Lun     9    3    4    5    3   16
```

**Key Observations**:
- Control/CRC confusion: bidirectional (8 samples each way)
- CRC/Lung confusion: biologically plausible (both epithelial cancers)
- Gastric: most scattered across all predictions
- Liver: cleanest predictions despite smallest sample size

---

## 🔮 Future Improvements

1. **Increase Gastric sample size** (~50 → 100 samples)
2. **Add cancer-specific biomarkers** from external databases
3. **Deep learning** approaches for end-to-end feature learning
4. **Production deployment**: Flag samples with confidence <0.7 for manual review

---

## 📚 References

- cfDNA fragmentomics: Snyder et al. (2016) "Cell-free DNA Comprises an In Vivo Nucleosome Footprint"
- PLS-DA VIP: Chong & Jun (2005) "Performance of some variable selection methods"
- Stability Selection: Meinshausen & Bühlmann (2010) "Stability selection"
