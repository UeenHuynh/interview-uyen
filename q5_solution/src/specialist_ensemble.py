#!/usr/bin/env python3
"""
Specialist Ensemble Architecture

Uses the correct v3 model: voting_models_v3_notuned_xgb_rf.pkl

Architecture:
    INPUT (15 features - v3 Group PCA)
        ↓
    ┌─────────────────────────┐
    │   Voting Ensemble       │ ← 4 models (LogReg, SVM, RF, XGB) from v3
    │   (General Predictor)   │
    │   Output: 6-class proba │
    └─────────────────────────┘
        ↓
        probas_general (shape: n×6)
        ↓
    ┌──────────────┐  ┌──────────────┐
    │ CRC Binary   │  │ Gastric Bin  │
    │ Specialist   │  │ Specialist   │
    │ (CRC vs Rest)│  │(Gast vs Rest)│
    └──────────────┘  └──────────────┘
        ↓                  ↓
      p_crc            p_gastric
        ↓                  ↓
        └──────┬───────────┘
               ↓
        Probability Fusion:
        final[CRC] = α×general[CRC] + (1-α)×p_crc
        final[Gastric] = α×general[Gastric] + (1-α)×p_gastric
        final[others] = general[others]
               ↓
        Renormalize (sum to 1)
               ↓
        FINAL PREDICTION
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')
VERSIONS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/versions')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}
CRC_IDX = 2
GASTRIC_IDX = 3

# Alpha values to test
ALPHA_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]

print("=" * 70)
print("Specialist Ensemble: Voting + CRC/Gastric Specialists")
print("=" * 70)

# ============================================================
# Load Data and Model Parameters (v3)
# ============================================================
# Use v3 features
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

# Load correct model from v3_voting_base
V3_MODEL_PATH = VERSIONS_DIR / 'v3_voting_base' / 'voting_models_v3_notuned_xgb_rf.pkl'
with open(V3_MODEL_PATH, 'rb') as f:
    v3_models = pickle.load(f)

print(f"\nLoaded model: {V3_MODEL_PATH}")
print(f"Feature set: {v3_models.get('feature_set', 'v3')}")

# Extract model parameters from loaded models
lr_params = {k: v for k, v in v3_models['lr'].get_params().items()
             if k in ['C', 'class_weight', 'solver', 'multi_class', 'penalty']}
svm_params = {k: v for k, v in v3_models['svm'].get_params().items()
              if k in ['C', 'gamma', 'kernel', 'class_weight']}
rf_params = {k: v for k, v in v3_models['rf'].get_params().items()
             if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'class_weight']}
xgb_params = {k: v for k, v in v3_models['xgb'].get_params().items()
              if k in ['learning_rate', 'max_depth', 'n_estimators', 'colsample_bytree', 'subsample']}

print(f"\nData: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"\nModel params from v3_voting_base:")
print(f"  LR: C={lr_params.get('C')}")
print(f"  SVM: C={svm_params.get('C')}, gamma={svm_params.get('gamma')}")
print(f"  RF: n_estimators={rf_params.get('n_estimators')}, max_depth={rf_params.get('max_depth')}")
print(f"  XGB: lr={xgb_params.get('learning_rate')}, max_depth={xgb_params.get('max_depth')}, n_estimators={xgb_params.get('n_estimators')}")

# ============================================================
# Helper Functions
# ============================================================

def create_voting_models():
    """Create fresh instances of voting base models using v3 params"""
    return {
        'lr': LogisticRegression(
            C=lr_params.get('C', 0.1),
            class_weight='balanced',
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        'svm': SVC(
            C=svm_params.get('C', 1.0),
            gamma=svm_params.get('gamma', 'scale'),
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_STATE
        ),
        'rf': RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 200),
            max_depth=rf_params.get('max_depth', None),
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            min_samples_split=rf_params.get('min_samples_split', 2),
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'xgb': XGBClassifier(
            learning_rate=xgb_params.get('learning_rate', 0.1),
            max_depth=xgb_params.get('max_depth', 6),
            n_estimators=xgb_params.get('n_estimators', 200),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            subsample=xgb_params.get('subsample', 0.8),
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    }

def create_specialist(target_class):
    """Create binary classifier for target_class vs rest"""
    # Use XGBoost for specialists (best single model)
    return XGBClassifier(
        learning_rate=0.1,
        max_depth=3,  # Simpler for binary
        n_estimators=100,
        subsample=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        n_jobs=-1
    )

def train_voting(models, X, y, sample_weights=None):
    """Train all voting models"""
    for name, model in models.items():
        if name == 'xgb' and sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
    return models

def predict_voting_proba(models, X):
    """Get soft voting probabilities"""
    probas = []
    for model in models.values():
        probas.append(model.predict_proba(X))
    return np.mean(probas, axis=0)

def fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=0.5):
    """
    Fuse general predictor with specialists

    Args:
        general_proba: (n, 6) - 6-class probabilities from voting
        crc_proba: (n,) - P(CRC) from CRC specialist
        gastric_proba: (n,) - P(Gastric) from Gastric specialist
        alpha: weight for general predictor (1-alpha for specialist)

    Returns:
        (n, 6) - Fused and renormalized probabilities
    """
    fused = general_proba.copy()

    # Fuse CRC
    fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba

    # Fuse Gastric
    fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba

    # Renormalize to sum to 1
    fused = fused / fused.sum(axis=1, keepdims=True)

    return fused

# ============================================================
# Nested CV Evaluation - Test Multiple Alpha Values
# ============================================================
print("\n" + "=" * 70)
print("Running 5-Fold CV for Multiple Alpha Values")
print("=" * 70)
print(f"Alpha values to test: {ALPHA_VALUES}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Storage: alpha -> results
all_alpha_results = {}

# First, run CV to get voting probas and specialist probas (independent of alpha)
voting_only_scores = []
voting_only_preds = np.zeros(len(y_train), dtype=int)
general_probas_all = np.zeros((len(y_train), 6))
crc_probas_all = np.zeros(len(y_train))
gastric_probas_all = np.zeros(len(y_train))

print("\n" + "-" * 70)
print("Training models and collecting probabilities...")
print("-" * 70)

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\n--- Fold {fold_idx + 1}/5 ---")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    sample_weights = compute_sample_weight('balanced', y_tr)

    # ========== 1. Train Voting Ensemble ==========
    voting_models = create_voting_models()
    voting_models = train_voting(voting_models, X_tr, y_tr, sample_weights)

    # Get voting probabilities
    general_proba = predict_voting_proba(voting_models, X_val)
    voting_pred = np.argmax(general_proba, axis=1)
    general_probas_all[val_idx] = general_proba

    # ========== 2. Train CRC Specialist (CRC vs Rest) ==========
    y_crc_binary = (y_tr == CRC_IDX).astype(int)
    crc_weights = compute_sample_weight('balanced', y_crc_binary)

    crc_specialist = create_specialist(CRC_IDX)
    crc_specialist.fit(X_tr, y_crc_binary, sample_weight=crc_weights)
    crc_proba = crc_specialist.predict_proba(X_val)[:, 1]
    crc_probas_all[val_idx] = crc_proba

    # ========== 3. Train Gastric Specialist (Gastric vs Rest) ==========
    y_gastric_binary = (y_tr == GASTRIC_IDX).astype(int)
    gastric_weights = compute_sample_weight('balanced', y_gastric_binary)

    gastric_specialist = create_specialist(GASTRIC_IDX)
    gastric_specialist.fit(X_tr, y_gastric_binary, sample_weight=gastric_weights)
    gastric_proba = gastric_specialist.predict_proba(X_val)[:, 1]
    gastric_probas_all[val_idx] = gastric_proba

    # ========== 4. Evaluate Voting Only ==========
    voting_f1 = f1_score(y_val, voting_pred, average='macro')
    voting_only_scores.append(voting_f1)
    voting_only_preds[val_idx] = voting_pred

    print(f"  Voting F1={voting_f1:.3f}")

# Calculate voting baseline
voting_mean = np.mean(voting_only_scores)
voting_std = np.std(voting_only_scores)
voting_f1_class = f1_score(y_train, voting_only_preds, average=None)

print(f"\nVoting Baseline: F1={voting_mean:.3f} ± {voting_std:.3f}")

# ============================================================
# Test Each Alpha Value
# ============================================================
print("\n" + "=" * 70)
print("Testing Alpha Values")
print("=" * 70)

for alpha in ALPHA_VALUES:
    print(f"\n--- Alpha = {alpha} ---")

    # Fuse probabilities with this alpha
    fused_probas = fuse_probabilities(general_probas_all, crc_probas_all, gastric_probas_all, alpha=alpha)
    specialist_preds = np.argmax(fused_probas, axis=1)

    # Calculate per-fold scores
    specialist_fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        y_val = y_train[val_idx]
        fold_preds = specialist_preds[val_idx]
        fold_f1 = f1_score(y_val, fold_preds, average='macro')
        specialist_fold_scores.append(fold_f1)

    specialist_mean = np.mean(specialist_fold_scores)
    specialist_std = np.std(specialist_fold_scores)
    specialist_f1_class = f1_score(y_train, specialist_preds, average=None)

    all_alpha_results[alpha] = {
        'f1_macro_mean': specialist_mean,
        'f1_macro_std': specialist_std,
        'per_class': {LABEL_MAP[i]: float(specialist_f1_class[i]) for i in range(6)},
        'fold_scores': specialist_fold_scores,
        'improvement': specialist_mean - voting_mean
    }

    print(f"  F1={specialist_mean:.3f} ± {specialist_std:.3f} (Δ={specialist_mean - voting_mean:+.3f})")
    print(f"  CRC={specialist_f1_class[CRC_IDX]:.3f}, Gastric={specialist_f1_class[GASTRIC_IDX]:.3f}")

# ============================================================
# Results Summary - All Alpha Values
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY - ALL ALPHA VALUES")
print("=" * 70)

print(f"\n{'Alpha':<10} {'F1 macro':<18} {'CRC':<10} {'Gastric':<10} {'Δ vs Voting'}")
print("-" * 60)
print(f"{'Voting':<10} {voting_mean:.3f} ± {voting_std:.3f}    {voting_f1_class[CRC_IDX]:.3f}      {voting_f1_class[GASTRIC_IDX]:.3f}      baseline")
print("-" * 60)

best_alpha = None
best_f1 = 0
for alpha, res in sorted(all_alpha_results.items()):
    f1 = res['f1_macro_mean']
    crc = res['per_class']['CRC']
    gastric = res['per_class']['Gastric']
    improvement = res['improvement']
    marker = ""
    if f1 > best_f1:
        best_f1 = f1
        best_alpha = alpha
    print(f"α={alpha:<6}  {f1:.3f} ± {res['f1_macro_std']:.3f}    {crc:.3f}      {gastric:.3f}      {improvement:+.3f}")

print("-" * 60)
print(f"\n*** Best Alpha: {best_alpha} with F1={best_f1:.3f} ***")

# Find best alpha result for detailed analysis
best_result = all_alpha_results[best_alpha]
best_specialist_preds = np.argmax(
    fuse_probabilities(general_probas_all, crc_probas_all, gastric_probas_all, alpha=best_alpha),
    axis=1
)
best_specialist_f1_class = f1_score(y_train, best_specialist_preds, average=None)

# Per-class F1 comparison for best alpha
print("\n" + "-" * 70)
print(f"Per-class F1 (Best α={best_alpha}):")
print("-" * 70)

print(f"{'Class':<12} {'Voting':<10} {'+ Specialist':<12} {'Change':<10}")
print("-" * 45)
for i, name in LABEL_MAP.items():
    change = best_specialist_f1_class[i] - voting_f1_class[i]
    indicator = "↑" if change > 0.02 else ("↓" if change < -0.02 else "→")
    marker = "★" if i in [CRC_IDX, GASTRIC_IDX] else ""
    print(f"{name:<12} {voting_f1_class[i]:.3f}      {best_specialist_f1_class[i]:.3f}        "
          f"{change:+.3f} {indicator} {marker}")

# Statistical test for best alpha
print("\n" + "-" * 70)
print(f"Statistical Test (Paired t-test) for α={best_alpha}:")
print("-" * 70)

t_stat, p_value = stats.ttest_rel(best_result['fold_scores'], voting_only_scores)
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

# ============================================================
# Comparison with Voting Baseline
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: VOTING vs SPECIALIST ENSEMBLE")
print("=" * 70)

# Load v3 voting baseline results
with open(VERSIONS_DIR / 'v3_voting_base' / 'voting_results_v3_notuned_xgb_rf.json') as f:
    v3_voting_results = json.load(f)

print(f"\n{'Method':<25} {'F1 macro':<12} {'CRC':<10} {'Gastric':<10}")
print("-" * 60)
print(f"{'Voting v3 (baseline)':<25} {v3_voting_results['voting']['f1_macro_mean']:.3f}        "
      f"{v3_voting_results['voting']['per_class']['CRC']:.3f}      "
      f"{v3_voting_results['voting']['per_class']['Gastric']:.3f}")
print(f"{'+ Specialist (α=' + str(best_alpha) + ')':<25} {best_f1:.3f}        "
      f"{best_result['per_class']['CRC']:.3f}      "
      f"{best_result['per_class']['Gastric']:.3f}")

improvement = best_f1 - v3_voting_results['voting']['f1_macro_mean']
print(f"\nImprovement: {improvement:+.4f}")

# ============================================================
# Save Results
# ============================================================
print("\n" + "-" * 70)
print("Saving results...")
print("-" * 70)

results = {
    'architecture': 'Voting v3 + CRC/Gastric Specialists',
    'base_model': 'voting_models_v3_notuned_xgb_rf.pkl',
    'feature_set': 'v3',
    'cv_folds': 5,
    'alpha_values_tested': ALPHA_VALUES,
    'best_alpha': best_alpha,
    'voting_baseline': {
        'f1_macro_mean': float(voting_mean),
        'f1_macro_std': float(voting_std),
        'per_class': {LABEL_MAP[i]: float(voting_f1_class[i]) for i in range(6)},
        'fold_scores': [float(s) for s in voting_only_scores]
    },
    'alpha_results': {
        str(alpha): {
            'f1_macro_mean': float(res['f1_macro_mean']),
            'f1_macro_std': float(res['f1_macro_std']),
            'per_class': res['per_class'],
            'fold_scores': [float(s) for s in res['fold_scores']],
            'improvement': float(res['improvement'])
        }
        for alpha, res in all_alpha_results.items()
    },
    'best_result': {
        'alpha': best_alpha,
        'f1_macro_mean': float(best_f1),
        'f1_macro_std': float(best_result['f1_macro_std']),
        'per_class': best_result['per_class'],
        'improvement': float(best_result['improvement'])
    },
    'statistical_test': {
        'test': 'paired_t_test',
        'alpha_tested': best_alpha,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }
}

with open(RESULTS_DIR / 'specialist_ensemble_results_v3.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved: specialist_ensemble_results_v3.json")

# ============================================================
# Visualization
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Per-class comparison (best alpha)
ax1 = axes[0]
x = np.arange(6)
width = 0.35
bars1 = ax1.bar(x - width/2, voting_f1_class, width, label='Voting Only', color='#3498db')
bars2 = ax1.bar(x + width/2, best_specialist_f1_class, width, label=f'+ Specialists (α={best_alpha})', color='#2ecc71')
ax1.set_ylabel('F1 Score')
ax1.set_xticks(x)
ax1.set_xticklabels(list(LABEL_MAP.values()))
ax1.legend()
ax1.set_title(f'Per-class F1: Voting vs Specialist (α={best_alpha})')
ax1.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)

# Highlight CRC and Gastric
for i in [CRC_IDX, GASTRIC_IDX]:
    ax1.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='yellow')

# Add value labels
for bar, score in zip(bars1, voting_f1_class):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.2f}',
             ha='center', va='bottom', fontsize=8)
for bar, score in zip(bars2, best_specialist_f1_class):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.2f}',
             ha='center', va='bottom', fontsize=8)

# 2. Alpha comparison plot
ax2 = axes[1]
alphas = list(all_alpha_results.keys())
f1_means = [all_alpha_results[a]['f1_macro_mean'] for a in alphas]
f1_stds = [all_alpha_results[a]['f1_macro_std'] for a in alphas]

ax2.errorbar(alphas, f1_means, yerr=f1_stds, fmt='o-', color='#2ecc71', linewidth=2, markersize=8, capsize=5)
ax2.axhline(y=voting_mean, color='#3498db', linestyle='--', linewidth=2, label=f'Voting baseline ({voting_mean:.3f})')
ax2.set_xlabel('Alpha (α)')
ax2.set_ylabel('F1 Macro')
ax2.set_title('F1 Score vs Alpha Value')
ax2.legend()
ax2.set_ylim(voting_mean - 0.05, max(f1_means) + 0.05)

# Mark best alpha
best_idx = alphas.index(best_alpha)
ax2.scatter([best_alpha], [best_f1], color='red', s=100, zorder=5, marker='*')
ax2.annotate(f'Best: α={best_alpha}', xy=(best_alpha, best_f1), xytext=(best_alpha, best_f1 + 0.02),
             ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'specialist_ensemble_v3_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: specialist_ensemble_v3_comparison.png")

print("\n" + "=" * 70)
print("Specialist Ensemble Evaluation Complete!")
print(f"Best result: α={best_alpha}, F1={best_f1:.4f}")
print("=" * 70)