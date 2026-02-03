# Q5. Data Analysis and Machine Learning Model Construction (7 points)

## 📋 Overview

Complete machine learning pipeline for **cfDNA (cell-free DNA) fragmentomics classification** to detect cancer types from blood samples.

---

## 📁 File Structure

```
q5_solution/
├── README.md                    # This file
└── src/                         # Source code
    ├── stage0_prepare.py        # Data loading & preparation
    ├── stage1_qc.py             # Quality control & preprocessing
    ├── stage2_feature_selection.py  # Feature selection (VIP, stability)
    ├── stage2_group_pca_direct.py   # Group-aware PCA
    ├── stage3_phase1_baseline.py    # Baseline model training
    ├── stage3_phase2_tuning.py      # Hyperparameter tuning
    └── specialist_ensemble.py       # Final ensemble model
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

---

## 🔄 Pipeline Architecture

```
Stage 0: Data Preparation (stage0_prepare.py)
    • Load CSV files, transpose, extract labels
    ↓
Stage 1: Quality Control (stage1_qc.py)
    • Train/Test split (80/20, stratified)
    • Zero-variance filter → -9 features
    • Correlation filter (r>0.90) → -608 features
    • StandardScaler (fit on TRAIN only)
    └── Result: 1,158 → 541 features
    ↓
Stage 2: Feature Selection
    • stage2_feature_selection.py: VIP + Stability Selection
    • stage2_group_pca_direct.py: Group-aware PCA
    └── Result: 541 → 15 features (98.7% reduction)
    ↓
Stage 3: Model Training
    • stage3_phase1_baseline.py: Baseline models (LR, SVM, RF, XGB)
    • stage3_phase2_tuning.py: Hyperparameter tuning
    • specialist_ensemble.py: Final voting ensemble + specialists
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost
```

### Run the Pipeline
```bash
cd q5_solution/src

# 1. Data preparation
python stage0_prepare.py

# 2. Quality control
python stage1_qc.py

# 3. Feature selection
python stage2_feature_selection.py
python stage2_group_pca_direct.py

# 4. Model training
python stage3_phase1_baseline.py
python stage3_phase2_tuning.py

# 5. Final ensemble
python specialist_ensemble.py
```

---

## 📈 Results Summary

### Final Model: Voting + Specialists
| Metric | Value |
|--------|-------|
| **F1 Macro** | **0.475 ± 0.044** |
| Accuracy | 0.471 ± 0.047 |
| **AUC (macro)** | **0.794 ± 0.026** |

### Per-Class F1 Scores
| Class | F1 Score |
|-------|----------|
| Control | 0.510 |
| Breast | 0.480 |
| CRC | 0.458 |
| Gastric | 0.378 |
| Liver | 0.638 |
| Lung | 0.419 |

---

## 🔬 Key Findings

1. **Feature Reduction**: 98.7% (1,158 → 15 features)
2. **Best Performer**: Liver class (F1=0.64) despite smallest sample size
3. **Hardest Class**: Gastric (F1=0.38) - improved 12% with specialists
4. **Model Stability**: std reduced by 32% with specialist ensemble
