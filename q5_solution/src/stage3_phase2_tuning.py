#!/usr/bin/env python3
"""
Stage 3 - Phase 2: Hyperparameter Tuning

Models to tune:
1. Logistic Regression (best baseline: 0.423)
2. SVM Linear (baseline: 0.389)
3. SVM RBF (baseline: 0.419) - Added based on Phase 1 results

Using v2 features: Direct Group PCA (15D)
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}

# ============================================================
# Load Data
# ============================================================
print("=" * 70)
print("Stage 3 - Phase 2: Hyperparameter Tuning")
print("=" * 70)

# Feature set selection (v2 default, v3 uses *_v3.parquet)
FEATURE_SET = os.environ.get('FEATURE_SET', 'v2').lower()
FEATURE_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''
RESULTS_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''

X_train = pd.read_parquet(PROCESSED_DIR / f'X_train_final{FEATURE_SUFFIX}.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nData: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"Features: {FEATURE_SET} - {X_train.shape[1]} dimensions")

# CV setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# ============================================================
# Define Models and Parameter Grids
# ============================================================
print("\n" + "-" * 70)
print("Parameter Grids:")
print("-" * 70)

models_params = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                    solver='lbfgs', multi_class='multinomial'),
        'params': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'class_weight': [None, 'balanced']
        }
    },
    'SVM_Linear': {
        'model': SVC(kernel='linear', random_state=RANDOM_STATE),
        'params': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'class_weight': [None, 'balanced']
        }
    },
    'SVM_RBF': {
        'model': SVC(kernel='rbf', random_state=RANDOM_STATE),
        'params': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            'class_weight': [None, 'balanced']
        }
    }
}

for name, config in models_params.items():
    n_combos = 1
    for k, v in config['params'].items():
        n_combos *= len(v)
    print(f"\n{name}:")
    for k, v in config['params'].items():
        print(f"  {k}: {v}")
    print(f"  Total combinations: {n_combos}")

# ============================================================
# Hyperparameter Tuning with Nested CV
# ============================================================
print("\n" + "=" * 70)
print("Running Hyperparameter Tuning (Nested CV)")
print("=" * 70)

results = {}

for model_name, config in models_params.items():
    print(f"\n{'-' * 70}")
    print(f"Tuning: {model_name}")
    print(f"{'-' * 70}")

    # Inner CV for hyperparameter selection
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=inner_cv,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )

    # Outer CV scores
    outer_scores = []
    outer_predictions = np.zeros(len(y_train), dtype=int)
    best_params_per_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Fit with inner CV
        grid_search.fit(X_tr, y_tr)

        # Predict on validation fold
        y_pred = grid_search.predict(X_val)
        outer_predictions[val_idx] = y_pred

        # Score
        fold_score = f1_score(y_val, y_pred, average='macro')
        outer_scores.append(fold_score)
        best_params_per_fold.append(grid_search.best_params_)

        print(f"  Fold {fold_idx+1}: F1={fold_score:.3f} | Best: {grid_search.best_params_}")

    # Compute per-class F1 from all predictions
    f1_per_class = f1_score(y_train, outer_predictions, average=None)

    # Final tuning on full data
    grid_search.fit(X_train, y_train)
    final_best_params = grid_search.best_params_

    results[model_name] = {
        'f1_macro_mean': np.mean(outer_scores),
        'f1_macro_std': np.std(outer_scores),
        'f1_per_class': f1_per_class.tolist(),
        'best_params': final_best_params,
        'params_per_fold': best_params_per_fold,
        'predictions': outer_predictions.tolist()
    }

    print(f"\n  Mean F1 (macro): {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
    print(f"  Final best params: {final_best_params}")
    print(f"  Per-class F1: {dict(zip(LABEL_MAP.values(), [f'{f:.3f}' for f in f1_per_class]))}")

# ============================================================
# Compare Results
# ============================================================
print("\n" + "=" * 70)
print("TUNING RESULTS COMPARISON")
print("=" * 70)

# Baseline results for comparison (from Phase 1)
baselines = {
    'LogisticRegression': {'f1_macro': 0.404, 'gastric': 0.261},
    'SVM_Linear': {'f1_macro': 0.389, 'gastric': 0.216},  # Approximate from original data
    'SVM_RBF': {'f1_macro': 0.419, 'gastric': 0.200}      # Approximate from original data
}

print(f"\n{'Model':<20} {'Baseline':<12} {'Tuned':<15} {'Change':<10} {'Gastric':<10}")
print("-" * 70)

for model_name, res in results.items():
    baseline = baselines.get(model_name, {'f1_macro': 0, 'gastric': 0})
    tuned = res['f1_macro_mean']
    change = tuned - baseline['f1_macro']
    gastric = res['f1_per_class'][3]

    indicator = "↑" if change > 0.01 else ("↓" if change < -0.01 else "→")
    print(f"{model_name:<20} {baseline['f1_macro']:.3f}        "
          f"{tuned:.3f} ± {res['f1_macro_std']:.3f}   {change:+.3f} {indicator}    {gastric:.3f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]['f1_macro_mean'])
print(f"\n🏆 Best tuned model: {best_model[0]}")
print(f"   F1 macro: {best_model[1]['f1_macro_mean']:.3f}")
print(f"   Best params: {best_model[1]['best_params']}")

# Per-class comparison
print("\n" + "-" * 70)
print("Per-class F1 (Tuned Models):")
print("-" * 70)
print(f"{'Class':<12}", end="")
for model_name in results.keys():
    print(f"{model_name:<18}", end="")
print()
print("-" * 70)

for i, class_name in LABEL_MAP.items():
    print(f"{class_name:<12}", end="")
    for model_name, res in results.items():
        f1 = res['f1_per_class'][i]
        print(f"{f1:.3f}             ", end="")
    print()

# Gastric focus
print("\n" + "-" * 70)
print("🎯 GASTRIC FOCUS:")
print("-" * 70)
for model_name, res in results.items():
    gastric = res['f1_per_class'][3]
    balanced = 'balanced' in str(res['best_params'].get('class_weight', ''))
    print(f"  {model_name:<20}: F1={gastric:.3f} | class_weight={'balanced' if balanced else 'None'}")

# ============================================================
# Save Results
# ============================================================
print("\n" + "-" * 70)
print("Saving results...")
print("-" * 70)

with open(RESULTS_DIR / f'phase2_tuning_results{RESULTS_SUFFIX}.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: phase2_tuning_results{RESULTS_SUFFIX}.json")

# Save best models
best_models = {}
for model_name, config in models_params.items():
    # Recreate model with best params
    model = config['model'].set_params(**results[model_name]['best_params'])
    model.fit(X_train, y_train)
    best_models[model_name] = model

with open(RESULTS_DIR / f'phase2_tuned_models{RESULTS_SUFFIX}.pkl', 'wb') as f:
    pickle.dump(best_models, f)
print(f"  Saved: phase2_tuned_models{RESULTS_SUFFIX}.pkl")

print("\n" + "=" * 70)
print("Phase 2 Complete!")
print("=" * 70)
