#!/usr/bin/env python3
"""
Option 6 Alpha Tuning: Find optimal fusion weight α

Tests multiple α values to find the best balance between
general voting predictor and binary specialists.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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

# Alpha values to test
ALPHA_VALUES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

print("=" * 70)
print("Option 6: Alpha Tuning for Probability Fusion")
print("=" * 70)

# ============================================================
# Load Data
# ============================================================
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nData: {X_train.shape[0]} samples × {X_train.shape[1]} features")
print(f"Testing α values: {ALPHA_VALUES}")

# Model params
xgb_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
rf_params = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}

phase2_tuned_path = RESULTS_DIR / 'phase2_tuning_results_v3.json'
if phase2_tuned_path.exists():
    with open(phase2_tuned_path) as f:
        phase2_results = json.load(f)
    lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', {})
    svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', {'C': 1.0, 'gamma': 'scale'})
else:
    lr_params = {}
    svm_params = {'C': 1.0, 'gamma': 'scale'}

# ============================================================
# Helper Functions
# ============================================================

def create_voting_models():
    lr_kwargs = dict(lr_params)
    lr_kwargs.setdefault('class_weight', 'balanced')
    svm_kwargs = dict(svm_params)
    svm_kwargs.setdefault('class_weight', 'balanced')
    
    return {
        'lr': LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE),
        'svm': SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True),
        'rf': RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        'xgb': XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
    }

def create_specialist():
    return XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, subsample=0.8, 
                        random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)

def train_voting(models, X, y, sample_weights=None):
    for name, model in models.items():
        if name == 'xgb' and sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
    return models

def predict_voting_proba(models, X):
    probas = [model.predict_proba(X) for model in models.values()]
    return np.mean(probas, axis=0)

def fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=0.5):
    fused = general_proba.copy()
    fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba
    fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused

# ============================================================
# Evaluate Multiple Alpha Values
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {}

for alpha in ALPHA_VALUES:
    print(f"\n--- Testing α = {alpha} ---")
    
    specialist_scores = []
    crc_scores = []
    gastric_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        sample_weights = compute_sample_weight('balanced', y_tr)
        
        # Train Voting
        voting_models = create_voting_models()
        voting_models = train_voting(voting_models, X_tr, y_tr, sample_weights)
        general_proba = predict_voting_proba(voting_models, X_val)
        
        # Train CRC Specialist
        y_crc_binary = (y_tr == CRC_IDX).astype(int)
        crc_weights = compute_sample_weight('balanced', y_crc_binary)
        crc_specialist = create_specialist()
        crc_specialist.fit(X_tr, y_crc_binary, sample_weight=crc_weights)
        crc_proba = crc_specialist.predict_proba(X_val)[:, 1]
        
        # Train Gastric Specialist
        y_gastric_binary = (y_tr == GASTRIC_IDX).astype(int)
        gastric_weights = compute_sample_weight('balanced', y_gastric_binary)
        gastric_specialist = create_specialist()
        gastric_specialist.fit(X_tr, y_gastric_binary, sample_weight=gastric_weights)
        gastric_proba = gastric_specialist.predict_proba(X_val)[:, 1]
        
        # Fuse
        fused_proba = fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=alpha)
        specialist_pred = np.argmax(fused_proba, axis=1)
        
        # Evaluate
        specialist_f1 = f1_score(y_val, specialist_pred, average='macro')
        specialist_scores.append(specialist_f1)
        
        # Per-class
        specialist_f1_class = f1_score(y_val, specialist_pred, average=None)
        crc_scores.append(specialist_f1_class[CRC_IDX])
        gastric_scores.append(specialist_f1_class[GASTRIC_IDX])
    
    results[alpha] = {
        'f1_macro_mean': np.mean(specialist_scores),
        'f1_macro_std': np.std(specialist_scores),
        'crc_mean': np.mean(crc_scores),
        'gastric_mean': np.mean(gastric_scores),
        'fold_scores': specialist_scores
    }
    
    print(f"  F1: {results[alpha]['f1_macro_mean']:.3f} ± {results[alpha]['f1_macro_std']:.3f}")
    print(f"  CRC: {results[alpha]['crc_mean']:.3f}, Gastric: {results[alpha]['gastric_mean']:.3f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALPHA TUNING SUMMARY")
print("=" * 70)

print(f"\n{'α':<6} {'F1 Macro':<15} {'CRC':<10} {'Gastric':<10}")
print("-" * 45)

best_alpha = None
best_f1 = 0

for alpha in ALPHA_VALUES:
    r = results[alpha]
    marker = ""
    if r['f1_macro_mean'] > best_f1:
        best_f1 = r['f1_macro_mean']
        best_alpha = alpha
    print(f"{alpha:<6} {r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f}   {r['crc_mean']:.3f}      {r['gastric_mean']:.3f}")

print(f"\n🏆 Best α = {best_alpha} with F1 = {best_f1:.3f}")

# Note: α=1.0 means no specialist (pure voting)
voting_baseline = 0.473  # From previous run
print(f"\nVoting baseline: {voting_baseline:.3f}")
improvement = best_f1 - voting_baseline
print(f"Best improvement: {improvement:+.3f}")

# Save results
with open(RESULTS_DIR / 'option6_alpha_tuning.json', 'w') as f:
    json.dump({
        'alpha_values': ALPHA_VALUES,
        'results': {str(k): v for k, v in results.items()},
        'best_alpha': best_alpha,
        'best_f1': best_f1,
        'voting_baseline': voting_baseline,
        'improvement': improvement
    }, f, indent=2)
print("\n✓ Saved: option6_alpha_tuning.json")

# Plot
plt.figure(figsize=(10, 5))
alphas = list(results.keys())
f1s = [results[a]['f1_macro_mean'] for a in alphas]
stds = [results[a]['f1_macro_std'] for a in alphas]

plt.errorbar(alphas, f1s, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
plt.axhline(y=voting_baseline, color='red', linestyle='--', label=f'Voting baseline ({voting_baseline:.3f})')
plt.xlabel('Alpha (α)')
plt.ylabel('F1 Macro')
plt.title('Option 6: Alpha Tuning Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(RESULTS_DIR / 'option6_alpha_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: option6_alpha_tuning.png")
