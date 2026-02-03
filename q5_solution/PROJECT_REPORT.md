# Cancer Classification from cfDNA Fragmentomics

## Project Overview

**Objective**: Build machine learning models to classify 6 classes (1 healthy + 5 cancer types) from cell-free DNA (cfDNA) fragmentomics data.

**Dataset**: Gene Solutions Interview Test Q5 - Supplement_datasets

**Date**: February 1, 2026

---

## Table of Contents

1. [Dataset Description](#1-dataset-description)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 0: Data Preparation](#3-stage-0-data-preparation)
4. [Stage 1: Quality Control](#4-stage-1-quality-control)
5. [Stage 2: Feature Selection](#5-stage-2-feature-selection)
6. [Stage 3: Model Training](#6-stage-3-model-training)
7. [Results Summary](#7-results-summary)
8. [File Structure](#8-file-structure)
9. [How to Run](#9-how-to-run)

---

## 1. Dataset Description

### 1.1 Classes

| Class | Type | Samples | Percentage |
|-------|------|---------|------------|
| Control | Healthy | 70 | 23.3% |
| Breast | Cancer | 50 | 16.7% |
| CRC (Colorectal) | Cancer | 50 | 16.7% |
| Gastric | Cancer | 50 | 16.7% |
| Liver | Cancer | 30 | 10.0% |
| Lung | Cancer | 50 | 16.7% |
| **Total** | - | **300** | **100%** |

### 1.2 Feature Groups

| Feature Group | Description | Original Features |
|---------------|-------------|-------------------|
| **EM** | End Motif patterns (4-mer DNA sequences) | 256 |
| **FLEN** | Fragment Length distribution (50-350 bp) | 301 |
| **NUCLEOSOME** | Nucleosome positioning around TSS (-300 to +300) | 601 |
| **Total** | - | **1,158** |

### 1.3 Key Challenges

- **High dimensionality**: 1,158 features vs 300 samples (ratio 3.86:1)
- **Class imbalance**: Liver only has 30 samples (10%)
- **Feature redundancy**: NUCLEOSOME has 5,403 highly correlated pairs (>0.95)
- **Small sample size**: Risk of overfitting

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC FEATURE SELECTION PIPELINE            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 0: Data Preparation                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   EM.csv    │    │  FLEN.csv   │    │NUCLEOSOME   │             │
│  │  (256 feat) │    │ (301 feat)  │    │  (601 feat) │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  Transpose + Prefix  │                            │
│                 │  + Label Encoding    │                            │
│                 │    (1,158 features)  │                            │
│                 └──────────┬──────────┘                             │
│                            │                                        │
├────────────────────────────┼────────────────────────────────────────┤
│  Stage 1: Quality Control  │  (ALL DECISIONS ON TRAIN DATA ONLY)   │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │ Train/Test Split    │                             │
│                 │ (80/20, stratified) │                             │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │ Zero-Variance Filter│  -9 features                │
│                 │ (based on TRAIN)    │                             │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │ Correlation Filter  │  -608 features              │
│                 │ (threshold=0.90)    │                             │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  StandardScaler     │                             │
│                 │  (fit on TRAIN)     │                             │
│                 │    (541 features)   │                             │
│                 └──────────┬──────────┘                             │
│                            │                                        │
├────────────────────────────┼────────────────────────────────────────┤
│  Stage 2: Feature Selection│                                        │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  Layer A: PLS-DA    │                             │
│                 │  VIP Scoring        │  -> 120 features            │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  Layer B: Stability │                             │
│                 │  Selection (LASSO)  │  -> 38 features             │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  Layer C: Group PCA │                             │
│                 │  (per feature group)│  -> 14 features             │
│                 └──────────┬──────────┘                             │
│                            │                                        │
├────────────────────────────┼────────────────────────────────────────┤
│  Stage 3: Model Training   │                                        │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  Nested CV          │                             │
│                 │  (10-outer/5-inner) │                             │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                 ┌─────────────────────┐                             │
│                 │  Permutation Test   │                             │
│                 │  (1000 iterations)  │                             │
│                 └──────────┬──────────┘                             │
│                            ▼                                        │
│                    Final Model                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 0: Data Preparation

**Script**: `src/stage0_prepare.py`

### Actions:
1. Load raw CSV files (EM, FLEN, NUCLEOSOME)
2. Transpose data (samples as rows, features as columns)
3. Add prefixes to distinguish feature groups (EM_, FLEN_, NUC_)
4. Extract labels from sample names
5. Save to Parquet format for efficient storage

### Output:
- `data/processed/cleaned_data.parquet` (300 samples × 1,158 features)
- `data/processed/labels.parquet`
- `data/metadata/feature_groups.json`

---

## 4. Stage 1: Quality Control

**Script**: `src/stage1_qc.py`

### Critical: Data Leakage Prevention
All QC decisions are made on **TRAINING data only** to prevent data leakage.

### Steps:

| Step | Action | Features Removed | Remaining |
|------|--------|------------------|-----------|
| 1.1 | Train/Test Split (80/20, stratified) | - | 1,158 |
| 1.2 | Zero-Variance Filter (TRAIN only) | 9 | 1,149 |
| 1.3 | Correlation Filter (threshold=0.90) | 608 | 541 |
| 1.4 | StandardScaler (fit on TRAIN) | - | 541 |

### Feature Breakdown After QC:

| Group | Original | After QC | Removed |
|-------|----------|----------|---------|
| EM | 256 | 86 | 170 |
| FLEN | 301 | 50 | 251 |
| NUCLEOSOME | 601 | 405 | 196 |
| **Total** | **1,158** | **541** | **617** |

### Output:
- `data/processed/X_train_scaled.parquet`
- `data/processed/X_test_scaled.parquet`
- `data/processed/y_train.parquet`
- `data/processed/y_test.parquet`
- `data/metadata/scaler.pkl`
- `data/metadata/qc_report.json`

---

## 5. Stage 2: Feature Selection

**Script**: `src/stage2_feature_selection.py`

### Layer A: Sparse PLS-DA with VIP Scoring

- **Method**: Partial Least Squares Discriminant Analysis
- **Purpose**: Supervised dimensionality reduction accounting for correlation structure
- **VIP (Variable Importance in Projection)**: Identifies features driving discrimination
- **Result**: 541 → 120 features (top VIP scores)

### Layer B: Stability Selection

- **Method**: LASSO on 100 bootstrap subsamples (50% each)
- **Threshold**: Features selected in ≥30% of iterations
- **Purpose**: Filter noise-driven features, reduce false discoveries
- **Result**: 120 → 38 features

### Layer C: Group-Aware PCA

Apply PCA independently to each biological group to preserve semantic meaning:

| Group | Stable Features | PCA Components | Variance Explained |
|-------|-----------------|----------------|-------------------|
| EM | 4 | 4 | 100.0% |
| FLEN | 6 | 5 | 97.0% |
| NUCLEOSOME | 28 | 5 | 66.5% |
| **Total** | **38** | **14** | - |

### Final Features:
```
EM_PC1, EM_PC2, EM_PC3, EM_PC4
FLEN_PC1, FLEN_PC2, FLEN_PC3, FLEN_PC4, FLEN_PC5
NUCLEOSOME_PC1, NUCLEOSOME_PC2, NUCLEOSOME_PC3, NUCLEOSOME_PC4, NUCLEOSOME_PC5
```

### Output:
- `data/processed/X_train_final.parquet` (240 × 14)
- `data/processed/X_test_final.parquet` (60 × 14)
- `data/processed/vip_scores.csv`
- `data/processed/stability_selection.csv`
- `data/metadata/pca_models.pkl`
- `data/metadata/selected_features.json`

---

## 6. Stage 3: Model Training

**Script**: `src/stage3_model.py` (To be implemented)

### Nested Cross-Validation Structure

| Loop | Folds | Purpose |
|------|-------|---------|
| Outer | 10-fold Stratified | Unbiased performance estimation |
| Inner | 5-fold Stratified | Hyperparameter tuning |

### Models to Evaluate

| Model | Key Constraints (Anti-overfit) |
|-------|-------------------------------|
| Logistic Regression (L2) | C ∈ {0.001, 0.01, 0.1, 1} |
| Random Forest | max_depth ≤ 5, min_samples_leaf ≥ 10 |
| Linear SVM | C ∈ {0.001, 0.01, 0.1, 1} |
| XGBoost | max_depth ≤ 5, n_estimators ≤ 200 |

### Classification Strategies

**Strategy 1: Two-Stage**
1. Stage 1: Control vs Disease (binary)
2. Stage 2: Cancer type classification (5-class)

**Strategy 2: Direct 6-Class**
- Single multi-class classifier

### Permutation Test
- 1000 label shuffles
- Validates statistical significance (p < 0.05)

---

## 7. Results Summary

### Feature Reduction Pipeline

```
Original:        1,158 features
After QC:          541 features  (-53.3%)
After VIP:         120 features  (-77.8%)
After Stability:    38 features  (-68.3%)
After Group PCA:    14 features  (-63.2%)

Total reduction: 1,158 → 14 features (98.8% removed)
```

### Data Split

| Set | Samples | Purpose |
|-----|---------|---------|
| Train | 240 (80%) | Model training & CV |
| Test | 60 (20%) | Final evaluation |

### Quality Assurance

| Check | Status |
|-------|--------|
| No data leakage | ✅ PASS |
| n/2 rule (features < n/2) | ✅ PASS (14 < 120) |
| Stratification preserved | ✅ PASS |
| No sample overlap | ✅ PASS |
| No missing values | ✅ PASS |

---

## 8. File Structure

```
q5_solution/
├── data/
│   ├── processed/
│   │   ├── cleaned_data.parquet      # Stage 0 output
│   │   ├── labels.parquet            # Labels
│   │   ├── qc_cleaned.parquet        # After QC (before scaling)
│   │   ├── X_train_scaled.parquet    # Stage 1 output (train)
│   │   ├── X_test_scaled.parquet     # Stage 1 output (test)
│   │   ├── X_train_final.parquet     # Stage 2 output (train)
│   │   ├── X_test_final.parquet      # Stage 2 output (test)
│   │   ├── y_train.parquet           # Train labels
│   │   ├── y_test.parquet            # Test labels
│   │   ├── vip_scores.csv            # PLS-DA VIP scores
│   │   └── stability_selection.csv   # Stability selection results
│   └── metadata/
│       ├── feature_groups.json       # Feature group mapping
│       ├── preprocessing_config.json # Config for reproducibility
│       ├── qc_report.json            # QC summary
│       ├── scaler.pkl                # Fitted StandardScaler
│       ├── pca_models.pkl            # Fitted PCA models
│       └── selected_features.json    # Final feature list
├── src/
│   ├── stage0_prepare.py             # Data preparation
│   ├── stage1_qc.py                  # Quality control
│   ├── stage2_feature_selection.py   # Feature selection
│   └── stage3_model.py               # Model training (TBD)
├── out/                              # EDA visualizations
│   ├── class_counts.png
│   ├── pca_flen.png
│   ├── pca_em.png
│   ├── pca_nucleosome.png
│   └── pca_combined.png
├── PROJECT_REPORT.md                 # This file
├── DATA_DISCOVERY.md                 # Data discovery report
└── README.md                         # Quick start guide
```

---

## 9. How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn pyarrow
```

### Execute Pipeline

```bash
cd q5_solution

# Stage 0: Prepare data
python src/stage0_prepare.py

# Stage 1: Quality control
python src/stage1_qc.py

# Stage 2: Feature selection
python src/stage2_feature_selection.py

# Stage 3: Model training (TBD)
python src/stage3_model.py
```

### Load Final Data for Modeling

```python
import pandas as pd

# Load processed features
X_train = pd.read_parquet('data/processed/X_train_final.parquet')
X_test = pd.read_parquet('data/processed/X_test_final.parquet')
y_train = pd.read_parquet('data/processed/y_train.parquet')['label']
y_test = pd.read_parquet('data/processed/y_test.parquet')['label']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Output: Train: (240, 14), Test: (60, 14)
```

---

## Expected Performance Targets

| Metric | Conservative | Realistic | Optimistic |
|--------|-------------|-----------|------------|
| Overall Accuracy | 65-70% | 75-80% | 85-90% |
| Macro F1-Score | 0.60-0.65 | 0.70-0.75 | 0.80-0.85 |
| Control F1 | 0.75-0.80 | 0.85-0.90 | 0.90-0.95 |
| Cancer Avg F1 | 0.55-0.60 | 0.65-0.70 | 0.75-0.80 |
| Liver F1 (hardest) | 0.40-0.50 | 0.55-0.65 | 0.70-0.75 |

---

## Critical Rules Followed

| # | Rule | Why |
|---|------|-----|
| 1 | Train/Test split FIRST | Prevents data leakage |
| 2 | Never fit on test data | All transformers fit on TRAIN only |
| 3 | Always stratified splits | Preserves class ratio (critical for Liver n=30) |
| 4 | Features < n/2 | Scientific guideline for p >> n settings |
| 5 | Save all fitted objects | Reproducibility - new data can be transformed identically |

---

**Author**: Claude Code
**Version**: 1.0
**Last Updated**: February 1, 2026
