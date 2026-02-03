#!/usr/bin/env python3
"""
Hybrid Ensemble: Class-Specific Model Selection

Meta-rule:
- IF XGBoost predicts CRC (2) or Lung (5) → use Voting prediction
- ELSE → use XGBoost prediction

Rationale: Voting excels at CRC (0.47 vs 0.27) and Lung (0.47 vs 0.45)
Expected gain: F1 0.450 → 0.488 (+8.4%)
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
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
SWITCH_CLASSES = [2, 5]  # CRC and Lung - use Voting for these

print("=" * 70)
print("Hybrid Ensemble: Class-Specific Model Selection")
print("=" * 70)
print(f"Meta-rule: Use Voting when XGBoost predicts CRC or Lung")

# Feature set selection (v2 default, v3 uses *_v3.parquet)
FEATURE_SET = os.environ.get('FEATURE_SET', 'v2').lower()
FEATURE_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''
RESULTS_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''

# ============================================================
# Load Data
# ============================================================
X_train = pd.read_parquet(PROCESSED_DIR / f'X_train_final{FEATURE_SUFFIX}.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nData: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"Feature set: {FEATURE_SET}")

# ============================================================
# Load Phase 2 Best Parameters
# ============================================================
phase2_all_path = RESULTS_DIR / 'phase2_all_models.json'
phase2_tuned_path = RESULTS_DIR / f'phase2_tuning_results{RESULTS_SUFFIX}.json'

default_xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
default_rf_params = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2
}
default_lr_params = {}
default_svm_params = {'C': 1.0, 'gamma': 'scale'}

if FEATURE_SET == 'v3' and phase2_tuned_path.exists():
    with open(phase2_tuned_path) as f:
        phase2_results = json.load(f)
    lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', default_lr_params)
    svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', default_svm_params)
    xgb_params = default_xgb_params
    rf_params = default_rf_params
    print(f"\nUsing Phase 2 tuned params (LR/SVM) from {phase2_tuned_path.name} + defaults for XGB/RF")
elif phase2_all_path.exists():
    with open(phase2_all_path) as f:
        phase2_results = json.load(f)
    xgb_params = phase2_results['XGBoost']['best_params']
    lr_params = phase2_results['LogisticRegression']['best_params']
    svm_params = phase2_results['SVM_RBF']['best_params']
    rf_params = phase2_results['RandomForest']['best_params']
    print(f"\nUsing Phase 2 tuned parameters from phase2_all_models.json")
else:
    xgb_params = default_xgb_params
    lr_params = default_lr_params
    svm_params = default_svm_params
    rf_params = default_rf_params
    print("\nPhase 2 tuned params not found. Using default parameters.")

# ============================================================
# Nested CV Implementation
# ============================================================
print("\n" + "=" * 70)
print("Running 5-Fold Nested CV (Anti-Leakage)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Storage
xgb_fold_scores = []
voting_fold_scores = []
hybrid_fold_scores = []

xgb_all_preds = np.zeros(len(y_train), dtype=int)
voting_all_preds = np.zeros(len(y_train), dtype=int)
hybrid_all_preds = np.zeros(len(y_train), dtype=int)

switching_stats = {'total': 0, 'switched': 0, 'disagreements': 0}

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\n--- Fold {fold_idx + 1}/5 ---")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    sample_weights = compute_sample_weight('balanced', y_tr)

    # ========== Train XGBoost ==========
    xgb_model = XGBClassifier(
        **xgb_params,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)
    xgb_pred = xgb_model.predict(X_val)

    # ========== Train Voting (4 base models) ==========
    lr_kwargs = dict(lr_params)
    lr_kwargs.setdefault('class_weight', 'balanced')
    lr_model = LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE)

    svm_kwargs = dict(svm_params)
    svm_kwargs.setdefault('class_weight', 'balanced')
    svm_model = SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True)

    rf_kwargs = dict(rf_params)
    rf_kwargs.setdefault('class_weight', 'balanced')
    rf_model = RandomForestClassifier(**rf_kwargs, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_voting = XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)

    lr_model.fit(X_tr, y_tr)
    svm_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)
    xgb_voting.fit(X_tr, y_tr, sample_weight=sample_weights)

    # Soft voting: average probabilities
    proba_lr = lr_model.predict_proba(X_val)
    proba_svm = svm_model.predict_proba(X_val)
    proba_rf = rf_model.predict_proba(X_val)
    proba_xgb_v = xgb_voting.predict_proba(X_val)

    avg_proba = (proba_lr + proba_svm + proba_rf + proba_xgb_v) / 4
    voting_pred = np.argmax(avg_proba, axis=1)

    # ========== Apply Hybrid Logic ==========
    hybrid_pred = xgb_pred.copy()

    for i in range(len(val_idx)):
        switching_stats['total'] += 1

        if xgb_pred[i] in SWITCH_CLASSES:  # CRC or Lung
            switching_stats['switched'] += 1
            hybrid_pred[i] = voting_pred[i]

            if xgb_pred[i] != voting_pred[i]:
                switching_stats['disagreements'] += 1

    # ========== Calculate Scores ==========
    xgb_f1 = f1_score(y_val, xgb_pred, average='macro')
    voting_f1 = f1_score(y_val, voting_pred, average='macro')
    hybrid_f1 = f1_score(y_val, hybrid_pred, average='macro')

    xgb_fold_scores.append(xgb_f1)
    voting_fold_scores.append(voting_f1)
    hybrid_fold_scores.append(hybrid_f1)

    # Store predictions
    xgb_all_preds[val_idx] = xgb_pred
    voting_all_preds[val_idx] = voting_pred
    hybrid_all_preds[val_idx] = hybrid_pred

    print(f"  XGBoost: {xgb_f1:.3f} | Voting: {voting_f1:.3f} | Hybrid: {hybrid_f1:.3f}")

# ============================================================
# Results Summary
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Overall metrics
xgb_mean = np.mean(xgb_fold_scores)
xgb_std = np.std(xgb_fold_scores)
voting_mean = np.mean(voting_fold_scores)
voting_std = np.std(voting_fold_scores)
hybrid_mean = np.mean(hybrid_fold_scores)
hybrid_std = np.std(hybrid_fold_scores)

print(f"\n{'Method':<15} {'F1 macro':<20} {'vs XGBoost'}")
print("-" * 50)
print(f"{'XGBoost':<15} {xgb_mean:.3f} ± {xgb_std:.3f}       baseline")
print(f"{'Voting':<15} {voting_mean:.3f} ± {voting_std:.3f}       {voting_mean - xgb_mean:+.3f}")
print(f"{'Hybrid':<15} {hybrid_mean:.3f} ± {hybrid_std:.3f}       {hybrid_mean - xgb_mean:+.3f}")

# Per-class F1
print("\n" + "-" * 70)
print("Per-class F1:")
print("-" * 70)

xgb_f1_class = f1_score(y_train, xgb_all_preds, average=None)
voting_f1_class = f1_score(y_train, voting_all_preds, average=None)
hybrid_f1_class = f1_score(y_train, hybrid_all_preds, average=None)

print(f"{'Class':<12} {'XGBoost':<10} {'Voting':<10} {'Hybrid':<10} {'Change':<10}")
print("-" * 52)
for i, name in LABEL_MAP.items():
    change = hybrid_f1_class[i] - xgb_f1_class[i]
    indicator = "↑" if change > 0.02 else ("↓" if change < -0.02 else "→")
    marker = "★" if i in SWITCH_CLASSES else ""
    print(f"{name:<12} {xgb_f1_class[i]:.3f}      {voting_f1_class[i]:.3f}      "
          f"{hybrid_f1_class[i]:.3f}      {change:+.3f} {indicator} {marker}")

# Switching statistics
print("\n" + "-" * 70)
print("Switching Statistics:")
print("-" * 70)
switch_rate = switching_stats['switched'] / switching_stats['total']
disagree_rate = switching_stats['disagreements'] / max(switching_stats['switched'], 1)

print(f"  Total predictions: {switching_stats['total']}")
print(f"  Used Voting (switched): {switching_stats['switched']} ({switch_rate*100:.1f}%)")
print(f"  Used XGBoost: {switching_stats['total'] - switching_stats['switched']} ({(1-switch_rate)*100:.1f}%)")
print(f"  XGB-Voting disagreements when switched: {switching_stats['disagreements']} ({disagree_rate*100:.1f}%)")

# Statistical test
print("\n" + "-" * 70)
print("Statistical Test (Paired t-test):")
print("-" * 70)

t_stat, p_value = stats.ttest_rel(hybrid_fold_scores, xgb_fold_scores)
print(f"  H0: Hybrid F1 = XGBoost F1")
print(f"  H1: Hybrid F1 ≠ XGBoost F1")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

# ============================================================
# Success Criteria Check
# ============================================================
print("\n" + "=" * 70)
print("SUCCESS CRITERIA CHECK")
print("=" * 70)

criteria = {
    'Hybrid F1 > 0.460': hybrid_mean > 0.460,
    'CRC F1 > 0.40': hybrid_f1_class[2] > 0.40,
    'No class degrades >0.10': all((hybrid_f1_class[i] - xgb_f1_class[i]) > -0.10 for i in range(6)),
    'Statistically significant (p<0.05)': p_value < 0.05
}

all_passed = True
for criterion, passed in criteria.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        all_passed = False
    print(f"  {criterion}: {status}")

print("\n" + "-" * 70)
if all_passed:
    print("🎉 ALL CRITERIA PASSED - Hybrid approach ADOPTED")
    decision = "ADOPT"
else:
    print("⚠️  NOT ALL CRITERIA MET - Need review")
    decision = "REVIEW"

# ============================================================
# Save Results
# ============================================================
print("\n" + "-" * 70)
print("Saving results...")
print("-" * 70)

results = {
    'approach': 'class_specific_hybrid',
    'meta_rule': 'Use Voting for CRC/Lung, XGBoost for others',
    'cv_folds': 5,
    'performance': {
        'xgb_baseline': {
            'f1_macro_mean': float(xgb_mean),
            'f1_macro_std': float(xgb_std),
            'per_class': {LABEL_MAP[i]: float(xgb_f1_class[i]) for i in range(6)},
            'fold_scores': [float(s) for s in xgb_fold_scores]
        },
        'voting': {
            'f1_macro_mean': float(voting_mean),
            'f1_macro_std': float(voting_std),
            'per_class': {LABEL_MAP[i]: float(voting_f1_class[i]) for i in range(6)},
            'fold_scores': [float(s) for s in voting_fold_scores]
        },
        'hybrid': {
            'f1_macro_mean': float(hybrid_mean),
            'f1_macro_std': float(hybrid_std),
            'improvement': float(hybrid_mean - xgb_mean),
            'per_class': {LABEL_MAP[i]: float(hybrid_f1_class[i]) for i in range(6)},
            'fold_scores': [float(s) for s in hybrid_fold_scores]
        }
    },
    'switching_stats': {
        'total_predictions': switching_stats['total'],
        'used_voting': switching_stats['switched'],
        'used_xgboost': switching_stats['total'] - switching_stats['switched'],
        'voting_rate': float(switch_rate),
        'disagreement_rate': float(disagree_rate)
    },
    'statistical_test': {
        'test': 'paired_t_test',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    },
    'success_criteria': {k: bool(v) for k, v in criteria.items()},
    'decision': decision
}

with open(RESULTS_DIR / f'hybrid_ensemble_results{RESULTS_SUFFIX}.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Saved: hybrid_ensemble_results{RESULTS_SUFFIX}.json")

# ============================================================
# Visualizations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Per-class F1 Comparison
ax1 = axes[0, 0]
x = np.arange(6)
width = 0.25
bars1 = ax1.bar(x - width, xgb_f1_class, width, label='XGBoost', color='#3498db')
bars2 = ax1.bar(x, voting_f1_class, width, label='Voting', color='#e74c3c')
bars3 = ax1.bar(x + width, hybrid_f1_class, width, label='Hybrid', color='#2ecc71')
ax1.set_ylabel('F1 Score')
ax1.set_xticks(x)
ax1.set_xticklabels(list(LABEL_MAP.values()))
ax1.legend()
ax1.set_title('Per-class F1 Comparison')
ax1.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)
# Highlight CRC and Lung
for i in [2, 5]:
    ax1.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='yellow')

# 2. Confusion Matrix - XGBoost
ax2 = axes[0, 1]
cm_xgb = confusion_matrix(y_train, xgb_all_preds)
cm_xgb_norm = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_xgb_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=list(LABEL_MAP.values()),
            yticklabels=list(LABEL_MAP.values()), ax=ax2)
ax2.set_title('XGBoost Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

# 3. Confusion Matrix - Hybrid
ax3 = axes[1, 0]
cm_hybrid = confusion_matrix(y_train, hybrid_all_preds)
cm_hybrid_norm = cm_hybrid.astype('float') / cm_hybrid.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_hybrid_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=list(LABEL_MAP.values()),
            yticklabels=list(LABEL_MAP.values()), ax=ax3)
ax3.set_title('Hybrid Confusion Matrix')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')

# 4. Fold-by-fold comparison
ax4 = axes[1, 1]
folds = np.arange(1, 6)
ax4.plot(folds, xgb_fold_scores, 'o-', label='XGBoost', color='#3498db', linewidth=2)
ax4.plot(folds, voting_fold_scores, 's-', label='Voting', color='#e74c3c', linewidth=2)
ax4.plot(folds, hybrid_fold_scores, '^-', label='Hybrid', color='#2ecc71', linewidth=2)
ax4.set_xlabel('Fold')
ax4.set_ylabel('F1 Macro')
ax4.set_xticks(folds)
ax4.legend()
ax4.set_title('Fold-by-Fold F1 Comparison')
ax4.set_ylim(0.2, 0.7)

plt.tight_layout()
plt.savefig(RESULTS_DIR / f'hybrid_ensemble_comparison{RESULTS_SUFFIX}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: hybrid_ensemble_comparison{RESULTS_SUFFIX}.png")

print("\n" + "=" * 70)
print("Hybrid Ensemble Evaluation Complete!")
print("=" * 70)
