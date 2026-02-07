# Q5. Data Analysis and Machine Learning Model Construction (7 points)

## 📋 Overview

Complete machine learning pipeline for **cfDNA (cell-free DNA) fragmentomics classification** to detect cancer types from blood samples.

---

## 📁 File Structure

```
q5_solution/
├── README.md                      # This file
└── src/
    ├── stage0_prepare.py          # Data loading & preparation
    ├── stage1_qc.py               # Quality control & preprocessing
    ├── stage2_feature_selection.py    # Feature selection (VIP, stability)
    ├── stage2_group_pca_direct.py     # Group-aware PCA
    ├── model_pipeline.py          # Unified model training pipeline
    └── model_pipeline_viz.py      # Model training with visualization
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
Stage 3: Model Training & Evaluation
    • model_pipeline.py: Individual models & ensembles
    • model_pipeline_viz.py: With visualization output
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost catboost matplotlib seaborn
```

### Run the Pipeline
```bash
cd q5_solution

# Run data pipeline (if data not preprocessed)
python src/stage0_prepare.py
python src/stage1_qc.py
python src/stage2_group_pca_direct.py

# Train & evaluate models
python src/model_pipeline.py --all --save
```

### Model Options
```bash
# Individual models
python src/model_pipeline.py --lr              # Logistic Regression
python src/model_pipeline.py --svm             # SVM
python src/model_pipeline.py --rf              # Random Forest
python src/model_pipeline.py --xgb             # XGBoost
python src/model_pipeline.py --catboost        # CatBoost

# Voting ensembles
python src/model_pipeline.py --voting              # LR+SVM+RF+XGB
python src/model_pipeline.py --voting-catboost     # LR+SVM+RF+CatBoost

# With Specialists (best performance)
python src/model_pipeline.py --voting-catboost-specialist --alpha 0.8
python src/model_pipeline.py --catboost-specialist --alpha 0.6

# With visualization output
python src/model_pipeline_viz.py --catboost-specialist --alpha 0.6
```

---

## 📈 Results Summary

### Best Model: Voting(CatBoost) + Specialists (α=0.8)
| Metric | Value |
|--------|-------|
| **F1 Macro** | **0.484 ± 0.055** |
| Accuracy | 0.475 |
| **AUC-ROC (macro)** | **0.793** |

### Model Comparison
| Model | F1 Macro |
|-------|----------|
| 🥇 Voting(CatBoost) + Specialists | **0.484** |
| 🥈 CatBoost + Specialists (α=0.6) | 0.482 |
| 🥉 Voting(LR+SVM+RF+CatBoost) | 0.484 |
| CatBoost only | 0.482 |
| Voting + Specialists | 0.460 |
| Voting (LR+SVM+RF+XGB) | 0.457 |

### Per-Class F1 Scores (Best Model)
| Class | F1 Score |
|-------|----------|
| Control | 0.504 |
| Breast | 0.416 |
| CRC | 0.481 ★ |
| Gastric | 0.410 ★ |
| Liver | 0.642 |
| Lung | 0.513 |

★ = Improved by specialists

---

## 🔬 Key Findings

1. **Feature Reduction**: 98.7% (1,158 → 15 features)
2. **Best Performer**: Liver class (F1=0.64) despite smallest sample size
3. **CatBoost Advantage**: CatBoost outperforms XGBoost (+2.7% F1)
4. **Specialist Improvement**: CRC and Gastric classes benefit from binary specialists
5. **Optimal Fusion**: α=0.8 for voting, α=0.6 for CatBoost base
