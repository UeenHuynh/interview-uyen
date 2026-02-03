#!/usr/bin/env python3
"""
Option 6: Voting Ensemble + Binary Specialists (CRC/Gastric)

Architecture:
    INPUT (15 features)
        ↓
    ┌─────────────────────────┐
    │   Voting Ensemble       │ ← 4 models (LR, SVM, RF, XGB)
    │   (General Predictor)   │
    │   Output: 6-class proba │
    └─────────────────────────┘
        ↓
        probas_general (n×6)
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

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
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
CRC_IDX = 2
GASTRIC_IDX = 3

# Fusion weight (α): how much to trust general predictor vs specialist
# Tuned via CV: α=0.8 gives best results (trust voting more, specialists boost)
ALPHA = 0.8

print("=" * 70)
print("Option 6: Voting Ensemble + CRC/Gastric Binary Specialists")
print("=" * 70)
print(f"Fusion weight α = {ALPHA} (general={ALPHA}, specialist={1-ALPHA})")

# ============================================================
# Load Data
# ============================================================
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nData: {X_train.shape[0]} samples × {X_train.shape[1]} features")

# Load voting model params (use defaults for non-tuned version)
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
rf_params = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Load LR/SVM tuned params
phase2_tuned_path = RESULTS_DIR / 'phase2_tuning_results_v3.json'
if phase2_tuned_path.exists():
    with open(phase2_tuned_path) as f:
        phase2_results = json.load(f)
    lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', {})
    svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', {'C': 1.0, 'gamma': 'scale'})
    print(f"Using tuned LR/SVM params from {phase2_tuned_path.name}")
else:
    lr_params = {}
    svm_params = {'C': 1.0, 'gamma': 'scale'}
    print("Using default LR/SVM params")

# ============================================================
# Helper Functions
# ============================================================

def create_voting_models():
    """Create fresh instances of the 4 voting base models"""
    lr_kwargs = dict(lr_params)
    lr_kwargs.setdefault('class_weight', 'balanced')
    
    svm_kwargs = dict(svm_params)
    svm_kwargs.setdefault('class_weight', 'balanced')
    
    return {
        'lr': LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE),
        'svm': SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True),
        'rf': RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, 
                                      class_weight='balanced', n_jobs=-1),
        'xgb': XGBClassifier(**xgb_params, random_state=RANDOM_STATE, 
                             eval_metric='mlogloss', n_jobs=-1)
    }


def create_specialist():
    """Create XGBoost binary classifier for specialist"""
    return XGBClassifier(
        max_depth=3,          # Simpler for binary task
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        n_jobs=-1
    )


def train_voting(models, X, y, sample_weights=None):
    """Train all 4 voting models"""
    for name, model in models.items():
        if name == 'xgb' and sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
    return models


def predict_voting_proba(models, X):
    """Get soft voting probabilities (average of 4 models)"""
    probas = []
    for model in models.values():
        probas.append(model.predict_proba(X))
    return np.mean(probas, axis=0)


def fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=0.5):
    """
    Fuse general predictor with specialists using probability fusion.
    
    Args:
        general_proba: (n, 6) - 6-class probabilities from voting
        crc_proba: (n,) - P(CRC) from CRC specialist
        gastric_proba: (n,) - P(Gastric) from Gastric specialist
        alpha: weight for general predictor (1-alpha for specialist)
    
    Returns:
        (n, 6) - Fused and renormalized probabilities
    """
    fused = general_proba.copy()
    
    # Fuse CRC probability
    fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba
    
    # Fuse Gastric probability
    fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba
    
    # Renormalize to ensure probabilities sum to 1
    fused = fused / fused.sum(axis=1, keepdims=True)
    
    return fused


# ============================================================
# 5-Fold Nested CV Evaluation
# ============================================================
print("\n" + "=" * 70)
print("Running 5-Fold Nested CV")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Storage for results
voting_only_scores = []
specialist_scores = []

voting_only_preds = np.zeros(len(y_train), dtype=int)
specialist_preds = np.zeros(len(y_train), dtype=int)
specialist_probas = np.zeros((len(y_train), 6))

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
    
    # ========== 2. Train CRC Specialist (CRC vs Rest) ==========
    y_crc_binary = (y_tr == CRC_IDX).astype(int)
    crc_weights = compute_sample_weight('balanced', y_crc_binary)
    
    crc_specialist = create_specialist()
    crc_specialist.fit(X_tr, y_crc_binary, sample_weight=crc_weights)
    
    # P(CRC) from specialist
    crc_proba = crc_specialist.predict_proba(X_val)[:, 1]
    
    # ========== 3. Train Gastric Specialist (Gastric vs Rest) ==========
    y_gastric_binary = (y_tr == GASTRIC_IDX).astype(int)
    gastric_weights = compute_sample_weight('balanced', y_gastric_binary)
    
    gastric_specialist = create_specialist()
    gastric_specialist.fit(X_tr, y_gastric_binary, sample_weight=gastric_weights)
    
    # P(Gastric) from specialist
    gastric_proba = gastric_specialist.predict_proba(X_val)[:, 1]
    
    # ========== 4. Fuse Probabilities ==========
    fused_proba = fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=ALPHA)
    specialist_pred = np.argmax(fused_proba, axis=1)
    
    # ========== 5. Evaluate ==========
    voting_f1 = f1_score(y_val, voting_pred, average='macro')
    specialist_f1 = f1_score(y_val, specialist_pred, average='macro')
    
    voting_only_scores.append(voting_f1)
    specialist_scores.append(specialist_f1)
    
    voting_only_preds[val_idx] = voting_pred
    specialist_preds[val_idx] = specialist_pred
    specialist_probas[val_idx] = fused_proba
    
    # Per-class breakdown
    voting_crc = f1_score(y_val == CRC_IDX, voting_pred == CRC_IDX)
    specialist_crc = f1_score(y_val == CRC_IDX, specialist_pred == CRC_IDX)
    voting_gastric = f1_score(y_val == GASTRIC_IDX, voting_pred == GASTRIC_IDX)
    specialist_gastric = f1_score(y_val == GASTRIC_IDX, specialist_pred == GASTRIC_IDX)
    
    print(f"  Voting:     F1={voting_f1:.3f} | CRC={voting_crc:.3f} | Gastric={voting_gastric:.3f}")
    print(f"  Specialist: F1={specialist_f1:.3f} | CRC={specialist_crc:.3f} | Gastric={specialist_gastric:.3f}")

# ============================================================
# Results Summary  
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

voting_mean = np.mean(voting_only_scores)
voting_std = np.std(voting_only_scores)
specialist_mean = np.mean(specialist_scores)
specialist_std = np.std(specialist_scores)

improvement = specialist_mean - voting_mean

print(f"\n{'Method':<20} {'F1 macro':<20} {'vs Voting'}")
print("-" * 55)
print(f"{'Voting Only':<20} {voting_mean:.3f} ± {voting_std:.3f}       baseline")
print(f"{'+ Specialists':<20} {specialist_mean:.3f} ± {specialist_std:.3f}       {improvement:+.3f}")

# Per-class F1
print("\n" + "-" * 70)
print("Per-class F1:")
print("-" * 70)

voting_f1_class = f1_score(y_train, voting_only_preds, average=None)
specialist_f1_class = f1_score(y_train, specialist_preds, average=None)

print(f"{'Class':<12} {'Voting':<10} {'+ Specialist':<12} {'Change':<10}")
print("-" * 45)
for i, name in LABEL_MAP.items():
    change = specialist_f1_class[i] - voting_f1_class[i]
    indicator = "↑" if change > 0.02 else ("↓" if change < -0.02 else "→")
    marker = "★" if i in [CRC_IDX, GASTRIC_IDX] else ""
    print(f"{name:<12} {voting_f1_class[i]:.3f}      {specialist_f1_class[i]:.3f}        "
          f"{change:+.3f} {indicator} {marker}")

# Statistical test
print("\n" + "-" * 70)
print("Statistical Test (Paired t-test):")
print("-" * 70)

t_stat, p_value = stats.ttest_rel(specialist_scores, voting_only_scores)
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant (p<0.05): {'Yes ✓' if p_value < 0.05 else 'No'}")

# ============================================================
# Save Results
# ============================================================
print("\n" + "-" * 70)
print("Saving results...")
print("-" * 70)

results = {
    'architecture': 'Voting + CRC/Gastric Binary Specialists (Option 6)',
    'fusion_weight_alpha': ALPHA,
    'cv_folds': 5,
    'base_model': 'voting_models_v3_notuned_xgb_rf.pkl',
    'performance': {
        'voting_only': {
            'f1_macro_mean': float(voting_mean),
            'f1_macro_std': float(voting_std),
            'per_class': {LABEL_MAP[i]: float(voting_f1_class[i]) for i in range(6)},
            'fold_scores': [float(s) for s in voting_only_scores]
        },
        'with_specialists': {
            'f1_macro_mean': float(specialist_mean),
            'f1_macro_std': float(specialist_std),
            'per_class': {LABEL_MAP[i]: float(specialist_f1_class[i]) for i in range(6)},
            'fold_scores': [float(s) for s in specialist_scores],
            'improvement': float(improvement)
        }
    },
    'statistical_test': {
        'test': 'paired_t_test',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }
}

with open(RESULTS_DIR / 'option6_specialist_voting_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  ✓ Saved: option6_specialist_voting_results.json")

# ============================================================
# Train Final Models on Full Data
# ============================================================
print("\nTraining final models on full data...")

# Voting models
final_voting = create_voting_models()
final_sample_weights = compute_sample_weight('balanced', y_train)
final_voting = train_voting(final_voting, X_train, y_train, final_sample_weights)

# CRC specialist
y_crc_full = (y_train == CRC_IDX).astype(int)
crc_weights_full = compute_sample_weight('balanced', y_crc_full)
final_crc = create_specialist()
final_crc.fit(X_train, y_crc_full, sample_weight=crc_weights_full)

# Gastric specialist
y_gastric_full = (y_train == GASTRIC_IDX).astype(int)
gastric_weights_full = compute_sample_weight('balanced', y_gastric_full)
final_gastric = create_specialist()
final_gastric.fit(X_train, y_gastric_full, sample_weight=gastric_weights_full)

final_models = {
    'voting': final_voting,
    'crc_specialist': final_crc,
    'gastric_specialist': final_gastric,
    'alpha': ALPHA,
    'feature_set': 'v3'
}

with open(RESULTS_DIR / 'option6_final_models.pkl', 'wb') as f:
    pickle.dump(final_models, f)
print("  ✓ Saved: option6_final_models.pkl")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Per-class comparison
ax1 = axes[0]
x = np.arange(6)
width = 0.35
bars1 = ax1.bar(x - width/2, voting_f1_class, width, label='Voting Only', color='#3498db')
bars2 = ax1.bar(x + width/2, specialist_f1_class, width, label='+ Specialists', color='#2ecc71')
ax1.set_ylabel('F1 Score')
ax1.set_xticks(x)
ax1.set_xticklabels(list(LABEL_MAP.values()))
ax1.legend()
ax1.set_title('Option 6: Per-class F1 Comparison')
ax1.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)

# Highlight CRC and Gastric
for i in [CRC_IDX, GASTRIC_IDX]:
    ax1.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='yellow')

# Add value labels
for bar, score in zip(bars1, voting_f1_class):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.2f}',
             ha='center', va='bottom', fontsize=8)
for bar, score in zip(bars2, specialist_f1_class):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.2f}',
             ha='center', va='bottom', fontsize=8)

# 2. Fold-by-fold comparison
ax2 = axes[1]
folds = np.arange(1, 6)
ax2.plot(folds, voting_only_scores, 'o-', label='Voting Only', color='#3498db', linewidth=2, markersize=8)
ax2.plot(folds, specialist_scores, 's-', label='+ Specialists', color='#2ecc71', linewidth=2, markersize=8)
ax2.set_xlabel('Fold')
ax2.set_ylabel('F1 Macro')
ax2.set_xticks(folds)
ax2.legend()
ax2.set_title('Fold-by-Fold F1 Comparison')
ax2.set_ylim(0.25, 0.65)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'option6_specialist_voting_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: option6_specialist_voting_comparison.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("Option 6 Evaluation Complete!")
print("=" * 70)

print(f"\n🏆 Final Result:")
print(f"   Voting Only: {voting_mean:.3f} ± {voting_std:.3f}")
print(f"   + Specialists: {specialist_mean:.3f} ± {specialist_std:.3f}")
print(f"   Improvement: {improvement:+.3f} ({improvement/voting_mean*100:+.1f}%)")

if improvement > 0:
    print("\n✅ Specialists IMPROVED performance!")
else:
    print("\n⚠️ Specialists did not improve performance.")
