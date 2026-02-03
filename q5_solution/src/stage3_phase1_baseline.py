#!/usr/bin/env python3
"""
Stage 3 - Phase 1: Baseline Models with Simple CV

Purpose:
- Establish performance "floor" with default hyperparameters
- Simple 5-fold Stratified CV (NOT nested CV)
- If any model < 40% F1, revisit feature selection
- Nested CV only needed in Phase 2 for accurate evaluation

Models:
- Logistic Regression (L2)
- Random Forest
- SVM (Linear & RBF)
- KNN
- XGBoost
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, make_scorer
)

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
PERFORMANCE_FLOOR = 0.40  # If F1 < 40%, need to revisit feature selection

np.random.seed(RANDOM_STATE)

# Label mapping
LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}

# ============================================================
# Load Data
# ============================================================
print("=" * 70)
print("Stage 3 - Phase 1: Baseline Models with Simple CV")
print("=" * 70)

# Feature set selection (v2 default, v3 uses *_v3.parquet)
FEATURE_SET = os.environ.get('FEATURE_SET', 'v2').lower()
FEATURE_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''
RESULTS_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''

X_train = pd.read_parquet(PROCESSED_DIR / f'X_train_final{FEATURE_SUFFIX}.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values
X_test = pd.read_parquet(PROCESSED_DIR / f'X_test_final{FEATURE_SUFFIX}.parquet')
y_test = pd.read_parquet(PROCESSED_DIR / 'y_test.parquet')['label'].values

print(f"\nData loaded:")
print(f"  Train: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} samples x {X_test.shape[1]} features")
print(f"  Feature set: {FEATURE_SET}")

print(f"\nTrain class distribution:")
for label, name in LABEL_MAP.items():
    count = (y_train == label).sum()
    print(f"  {name}: {count} ({count/len(y_train)*100:.1f}%)")

# ============================================================
# Define Baseline Models (Default Hyperparameters)
# ============================================================
print("\n" + "-" * 70)
print("Defining baseline models with DEFAULT hyperparameters...")
print("-" * 70)

models = {
    'Logistic Regression (L2)': LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        multi_class='multinomial'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'SVM (Linear)': SVC(
        kernel='linear',
        random_state=RANDOM_STATE,
        probability=True
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        random_state=RANDOM_STATE,
        probability=True
    ),
    'KNN (k=5)': KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),
}

# Add XGBoost if available
if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )

print(f"Models to evaluate: {len(models)}")
for name in models.keys():
    print(f"  - {name}")

# ============================================================
# Simple 5-Fold Stratified CV
# ============================================================
print("\n" + "-" * 70)
print(f"Running {N_FOLDS}-Fold Stratified Cross-Validation...")
print("-" * 70)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results = {}
cv_predictions = {}

for name, model in models.items():
    print(f"\n>>> {name}")

    # Multiple metrics
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    f1_macro_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    f1_weighted_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)

    # Get CV predictions for confusion matrix
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    cv_predictions[name] = y_pred_cv

    # Per-class F1
    f1_per_class = f1_score(y_train, y_pred_cv, average=None)

    results[name] = {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'f1_macro_mean': f1_macro_scores.mean(),
        'f1_macro_std': f1_macro_scores.std(),
        'f1_weighted_mean': f1_weighted_scores.mean(),
        'f1_weighted_std': f1_weighted_scores.std(),
        'f1_per_class': {LABEL_MAP[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))},
        'cv_scores': f1_macro_scores.tolist()
    }

    # Print results
    print(f"    Accuracy:   {accuracy_scores.mean():.3f} ± {accuracy_scores.std():.3f}")
    print(f"    F1 (macro): {f1_macro_scores.mean():.3f} ± {f1_macro_scores.std():.3f}")
    print(f"    F1 (weighted): {f1_weighted_scores.mean():.3f} ± {f1_weighted_scores.std():.3f}")
    print(f"    Per-class F1:")
    for i, (label, f1) in enumerate(zip(LABEL_MAP.values(), f1_per_class)):
        status = "⚠️" if f1 < PERFORMANCE_FLOOR else "✓"
        print(f"      {label}: {f1:.3f} {status}")

# ============================================================
# Results Summary Table
# ============================================================
print("\n" + "=" * 70)
print("BASELINE RESULTS SUMMARY")
print("=" * 70)

# Create summary DataFrame
summary_data = []
for name, res in results.items():
    summary_data.append({
        'Model': name,
        'Accuracy': f"{res['accuracy_mean']:.3f} ± {res['accuracy_std']:.3f}",
        'F1 (macro)': f"{res['f1_macro_mean']:.3f} ± {res['f1_macro_std']:.3f}",
        'F1 (weighted)': f"{res['f1_weighted_mean']:.3f} ± {res['f1_weighted_std']:.3f}",
        'Liver F1': f"{res['f1_per_class']['Liver']:.3f}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Sort by F1 macro
sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_macro_mean'], reverse=True)

print("\n" + "-" * 70)
print("Ranking by F1 (macro):")
print("-" * 70)
for rank, (name, res) in enumerate(sorted_models, 1):
    status = "✓ PASS" if res['f1_macro_mean'] >= PERFORMANCE_FLOOR else "⚠️ BELOW FLOOR"
    print(f"  {rank}. {name}: {res['f1_macro_mean']:.3f} {status}")

# ============================================================
# Performance Floor Check
# ============================================================
print("\n" + "=" * 70)
print(f"PERFORMANCE FLOOR CHECK (threshold = {PERFORMANCE_FLOOR*100:.0f}%)")
print("=" * 70)

below_floor = []
for name, res in results.items():
    if res['f1_macro_mean'] < PERFORMANCE_FLOOR:
        below_floor.append(name)

if below_floor:
    print(f"\n⚠️  WARNING: {len(below_floor)} model(s) below floor:")
    for name in below_floor:
        print(f"    - {name}: {results[name]['f1_macro_mean']:.3f}")
    print("\n    RECOMMENDATION: Revisit feature selection before Phase 2!")
else:
    print(f"\n✓ All models above {PERFORMANCE_FLOOR*100:.0f}% floor. Ready for Phase 2.")

# ============================================================
# Best Model Analysis
# ============================================================
best_name, best_res = sorted_models[0]
print("\n" + "-" * 70)
print(f"BEST BASELINE MODEL: {best_name}")
print("-" * 70)
print(f"  F1 (macro):    {best_res['f1_macro_mean']:.3f} ± {best_res['f1_macro_std']:.3f}")
print(f"  Accuracy:      {best_res['accuracy_mean']:.3f} ± {best_res['accuracy_std']:.3f}")
print(f"\n  Per-class F1 scores:")
for label, f1 in best_res['f1_per_class'].items():
    bar = "█" * int(f1 * 20)
    print(f"    {label:8s}: {f1:.3f} |{bar}")

# Confusion Matrix for best model
print(f"\n  Confusion Matrix (CV predictions):")
cm = confusion_matrix(y_train, cv_predictions[best_name])
cm_df = pd.DataFrame(cm,
                     index=[f"True_{LABEL_MAP[i]}" for i in range(6)],
                     columns=[f"Pred_{LABEL_MAP[i]}" for i in range(6)])
print(cm_df.to_string())

# ============================================================
# Test Set Evaluation (Quick Sanity Check)
# ============================================================
print("\n" + "=" * 70)
print("TEST SET SANITY CHECK (using best model)")
print("=" * 70)

# Train best model on full training set
best_model = models[best_name]
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_f1_per_class = f1_score(y_test, y_test_pred, average=None)

print(f"\n{best_name} on Test Set:")
print(f"  Accuracy:   {test_accuracy:.3f}")
print(f"  F1 (macro): {test_f1_macro:.3f}")

# Check for overfitting
cv_f1 = best_res['f1_macro_mean']
gap = cv_f1 - test_f1_macro
print(f"\n  CV vs Test gap: {gap:.3f}")
if abs(gap) > 0.10:
    print(f"  ⚠️  Gap > 10%: Potential overfitting detected!")
else:
    print(f"  ✓ Gap acceptable (< 10%)")

print(f"\n  Test Per-class F1:")
for i, (label, f1) in enumerate(zip(LABEL_MAP.values(), test_f1_per_class)):
    print(f"    {label}: {f1:.3f}")

# ============================================================
# Save Results
# ============================================================
print("\n" + "-" * 70)
print("Saving results...")
print("-" * 70)

# Save baseline results
baseline_report = {
    'phase': 'Phase 1 - Baseline',
    'cv_folds': N_FOLDS,
    'performance_floor': PERFORMANCE_FLOOR,
    'models': results,
    'ranking': [name for name, _ in sorted_models],
    'best_model': best_name,
    'best_f1_macro': float(best_res['f1_macro_mean']),
    'below_floor_models': below_floor,
    'test_sanity_check': {
        'model': best_name,
        'accuracy': float(test_accuracy),
        'f1_macro': float(test_f1_macro),
        'cv_test_gap': float(gap)
    }
}

with open(RESULTS_DIR / f'phase1_baseline_results{RESULTS_SUFFIX}.json', 'w') as f:
    json.dump(baseline_report, f, indent=2)
print(f"  Saved: phase1_baseline_results{RESULTS_SUFFIX}.json")

# Save summary CSV
summary_df.to_csv(RESULTS_DIR / f'phase1_baseline_summary{RESULTS_SUFFIX}.csv', index=False)
print(f"  Saved: phase1_baseline_summary{RESULTS_SUFFIX}.csv")

# Save best baseline model
with open(RESULTS_DIR / f'best_baseline_model{RESULTS_SUFFIX}.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"  Saved: best_baseline_model{RESULTS_SUFFIX}.pkl")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1 COMPLETE - BASELINE SUMMARY")
print("=" * 70)
print(f"""
Performance Floor:     {PERFORMANCE_FLOOR*100:.0f}%
Models Evaluated:      {len(models)}
Models Above Floor:    {len(models) - len(below_floor)}
Best Model:            {best_name}
Best CV F1 (macro):    {best_res['f1_macro_mean']:.3f}
Test F1 (macro):       {test_f1_macro:.3f}
CV-Test Gap:           {gap:.3f}

Next Steps:
""")

if below_floor:
    print("  ⚠️  Some models below floor. Consider:")
    print("     1. Revisit feature selection (Stage 2)")
    print("     2. Try different PCA components")
    print("     3. Check for data issues")
else:
    print("  ✓ All baselines acceptable. Proceed to Phase 2:")
    print("     1. Nested CV for unbiased evaluation")
    print("     2. Hyperparameter tuning")
    print("     3. Compare Strategy 1 vs Strategy 2")
