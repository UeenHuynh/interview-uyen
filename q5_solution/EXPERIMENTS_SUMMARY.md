# Q5 cfDNA Classification - Experiments Summary

**Project:** cfDNA Fragmentomics 6-Class Cancer Classification  
**Dataset:** 300 samples × 1158 features → 15 features (after Group PCA)  
**Classes:** Control (70), Breast (50), CRC (50), Gastric (50), Liver (30), Lung (50)  
**Evaluation:** 5-Fold Stratified Cross-Validation, F1 Macro (primary metric)

---

## 🏆 Best Model

| Metric | Value |
|--------|-------|
| **Model** | Voting Ensemble + Specialists (α=0.8) |
| **F1 Macro** | **0.475 ± 0.044** |
| **AUC** | **0.799 ± 0.025** |
| **Accuracy** | **0.471 ± 0.031** |

**Best Model Script:**
```
src/option6_specialist_voting.py
```

**Best Results File:**
```
results/option6_specialist_voting_results.json
```

---

## Experiments Overview

### Phase 1: Baseline Models

| Model | F1 Macro | File |
|-------|----------|------|
| Logistic Regression | 0.422 | `phase1_baseline_results_v3.json` |
| SVM (Linear) | 0.396 | `phase1_baseline_results_v3.json` |
| SVM (RBF) | 0.439 | `phase1_baseline_results_v3.json` |
| Random Forest | 0.441 | `phase1_baseline_results_v3.json` |
| XGBoost | 0.435 | `phase1_baseline_results_v3.json` |

---

### Phase 2: Hyperparameter Tuning

| Model | Before | After | Improvement | Best Params |
|-------|--------|-------|-------------|-------------|
| Logistic Regression | 0.422 | **0.455** | +0.033 | C=0.1, class_weight=balanced |
| SVM (RBF) | 0.439 | **0.469** | +0.030 | C=1.0, gamma=scale |
| XGBoost | 0.435 | 0.460 | +0.025 | max_depth=3, n_estimators=200 |
| Random Forest | 0.441 | 0.475 | +0.034 | max_depth=6, min_samples_leaf=2 |

**Files:**
- `phase2_tuning_results_v3.json` (LR, SVM)
- `xgb_rf_tuning_v3.json` (XGB, RF)

---

### Phase 3: Voting Ensemble

| Config | F1 Macro | Std | Gastric F1 | File |
|--------|----------|-----|------------|------|
| Voting (NON-TUNED XGB/RF) | 0.473 | 0.065 | 0.338 | `voting_results_v3_notuned_xgb_rf.json` |
| Voting (TUNED XGB/RF) | 0.472 | **0.034** | **0.384** | `voting_results_v3_tuned_xgb_rf.json` |

**Observation:** TUNED có std thấp hơn (ổn định hơn) nhưng NON-TUNED có F1 cao hơn một chút.

---

### Phase 4: Specialist Ensemble (Option 6)

**Architecture:** Voting + Binary Specialists (CRC + Gastric) with probability fusion

#### Alpha Tuning Results:

| α | F1 Macro | CRC F1 | Gastric F1 |
|---|----------|--------|------------|
| 0.3 | 0.439 | 0.389 | 0.342 |
| 0.5 | 0.447 | 0.416 | 0.317 |
| 0.6 | 0.450 | 0.425 | 0.332 |
| 0.7 | 0.454 | 0.433 | 0.328 |
| **0.8** | **0.475** | **0.448** | **0.372** |
| 0.9 | 0.472 | 0.455 | 0.343 |

**Best:** α = 0.8 (80% Voting + 20% Specialist)

**File:** `option6_alpha_tuning.json`

---

### Comparison: DEFAULT vs TUNED XGB/RF with Specialists

| Config | Best α | F1 Macro | CRC F1 | Gastric F1 |
|--------|--------|----------|--------|------------|
| **DEFAULT XGB/RF** | **0.8** | **0.475** | **0.448** | **0.372** |
| TUNED XGB/RF | 0.7 | 0.467 | 0.421 | 0.335 |

**Winner:** DEFAULT XGB/RF (hardcoded params)

**File:** `tuned_vs_default_comparison.json`

---

### Supplementary: Repeated K-Fold CV

| Metric | Value |
|--------|-------|
| Repeats × Folds | 5 × 5 = 25 |
| F1 Macro (mean) | 0.457 |
| F1 Macro (std) | 0.068 |
| 95% CI | [0.430, 0.483] |

**File:** `repeated_kfold_results.json`

---

## Final Model Configuration

```python
# Models in Voting Ensemble
LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)
SVC(C=1.0, gamma='scale', class_weight='balanced', probability=True)
RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced')
XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)

# Specialists
CRC_Specialist = XGBClassifier(max_depth=3, n_estimators=100)
Gastric_Specialist = XGBClassifier(max_depth=3, n_estimators=100)

# Fusion
α = 0.8
P_final = α × P_voting + (1-α) × P_specialist
```

---

## All Result Files

| File | Description |
|------|-------------|
| `phase1_baseline_results_v3.json` | Baseline model performance |
| `phase2_tuning_results_v3.json` | LR/SVM tuning results |
| `xgb_rf_tuning_v3.json` | XGB/RF tuning results |
| `voting_results_v3.json` | Initial voting results |
| `voting_results_v3_tuned_xgb_rf.json` | Voting with tuned XGB/RF |
| `voting_results_v3_notuned_xgb_rf.json` | Voting with default XGB/RF |
| `option6_alpha_tuning.json` | Alpha tuning for specialists |
| **`option6_specialist_voting_results.json`** | **Final best model results** |
| `tuned_vs_default_comparison.json` | Comparison analysis |
| `repeated_kfold_results.json` | Stability analysis |
| `final_metrics_with_auc.json` | AUC and accuracy metrics |
| `hybrid_ensemble_results_v3.json` | Hybrid ensemble experiments |
| `specialist_ensemble_results_v3.json` | Specialist-only experiments |

---

## Key Findings

1. **Feature Selection:** Group PCA (EM:5 + FLEN:5 + NUC:5 = 15D) preserves 80%+ variance while reducing dimensionality 77x

2. **Best Architecture:** Voting Ensemble + Binary Specialists with α=0.8 fusion

3. **Gastric is hardest:** Lowest F1 (0.34-0.38) due to overlap with other GI cancers

4. **Liver performs best:** F1=0.58-0.64 despite smallest sample size (n=30)

5. **Label noise:** ~3% samples have potential mislabels (documented in `dataquality.md`)

6. **DEFAULT > TUNED for ensemble:** Individual model tuning doesn't always improve ensemble performance

---

## How to Run

```bash
cd /home/neeyuhuynh/Desktop/me/genesolution/q5_solution

# Full pipeline
python3 src/stage0_prepare.py
python3 src/stage1_qc.py
python3 src/stage2_feature_selection_v3.py
python3 src/voting_only_v3.py
python3 src/option6_specialist_voting.py

# Just the best model
python3 src/option6_specialist_voting.py
```

---

**Last Updated:** 2026-02-02
