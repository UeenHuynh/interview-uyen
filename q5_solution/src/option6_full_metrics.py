#!/usr/bin/env python3
"""
Compute full metrics (F1, AUC, Accuracy) for Option 6: Voting + Specialists (α=0.8)
This is the BEST model configuration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
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

# Best alpha from tuning
ALPHA = 0.8

print("=" * 70)
print("Option 6: Voting + Specialists (α=0.8) - Full Metrics")
print("=" * 70)

# ============================================================
# Load Data
# ============================================================
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nData: {X_train.shape[0]} samples × {X_train.shape[1]} features")

# Load LR/SVM tuned params
phase2_tuned_path = RESULTS_DIR / 'phase2_tuning_results_v3.json'
with open(phase2_tuned_path) as f:
    phase2_results = json.load(f)
lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', {})
svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', {'C': 1.0, 'gamma': 'scale'})

# DEFAULT XGB/RF params (best for ensemble)
xgb_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
rf_params = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}

print(f"Alpha: {ALPHA}")

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

def fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=0.8):
    fused = general_proba.copy()
    fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba
    fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused

# ============================================================
# 5-Fold CV with Full Metrics
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_f1 = []
fold_acc = []
fold_auc = []

all_y_true = []
all_y_pred = []
all_y_proba = []

print("\nRunning 5-Fold CV...\n")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"--- Fold {fold_idx + 1}/5 ---")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    sample_weights = compute_sample_weight('balanced', y_tr)
    
    # Train Voting Models
    voting_models = create_voting_models()
    for name, model in voting_models.items():
        if name == 'xgb':
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
        else:
            model.fit(X_tr, y_tr)
    
    # Get voting probabilities
    probas = [model.predict_proba(X_val) for model in voting_models.values()]
    general_proba = np.mean(probas, axis=0)
    
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
    
    # Fuse probabilities
    fused_proba = fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=ALPHA)
    final_pred = np.argmax(fused_proba, axis=1)
    
    # Compute metrics
    f1 = f1_score(y_val, final_pred, average='macro')
    acc = accuracy_score(y_val, final_pred)
    
    # AUC (one-vs-rest)
    y_val_bin = label_binarize(y_val, classes=[0, 1, 2, 3, 4, 5])
    auc = roc_auc_score(y_val_bin, fused_proba, average='macro', multi_class='ovr')
    
    fold_f1.append(f1)
    fold_acc.append(acc)
    fold_auc.append(auc)
    
    all_y_true.extend(y_val)
    all_y_pred.extend(final_pred)
    all_y_proba.extend(fused_proba)
    
    print(f"  F1 Macro: {f1:.3f} | Accuracy: {acc:.3f} | AUC: {auc:.3f}")

# ============================================================
# Final Results
# ============================================================
print("\n" + "=" * 70)
print("OPTION 6 FINAL METRICS (Voting + Specialists, α=0.8)")
print("=" * 70)

f1_mean, f1_std = np.mean(fold_f1), np.std(fold_f1)
acc_mean, acc_std = np.mean(fold_acc), np.std(fold_acc)
auc_mean, auc_std = np.mean(fold_auc), np.std(fold_auc)

# 95% CI
f1_se = f1_std / np.sqrt(5)
acc_se = acc_std / np.sqrt(5)
auc_se = auc_std / np.sqrt(5)

print(f"\nF1 Macro:  {f1_mean:.4f} ± {f1_std:.4f}  (95% CI: [{f1_mean - 1.96*f1_se:.3f}, {f1_mean + 1.96*f1_se:.3f}])")
print(f"Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}  (95% CI: [{acc_mean - 1.96*acc_se:.3f}, {acc_mean + 1.96*acc_se:.3f}])")
print(f"AUC Macro: {auc_mean:.4f} ± {auc_std:.4f}  (95% CI: [{auc_mean - 1.96*auc_se:.3f}, {auc_mean + 1.96*auc_se:.3f}])")

# Per-class F1
print("\nPer-class F1:")
f1_class = f1_score(all_y_true, all_y_pred, average=None)
for i, name in LABEL_MAP.items():
    print(f"  {name}: {f1_class[i]:.3f}")

# Save results
results = {
    'model': 'Option 6: Voting + Specialists',
    'alpha': ALPHA,
    'cv_folds': 5,
    'metrics': {
        'f1_macro': {
            'mean': float(f1_mean), 
            'std': float(f1_std),
            'ci_95': [float(f1_mean - 1.96*f1_se), float(f1_mean + 1.96*f1_se)]
        },
        'accuracy': {
            'mean': float(acc_mean), 
            'std': float(acc_std),
            'ci_95': [float(acc_mean - 1.96*acc_se), float(acc_mean + 1.96*acc_se)]
        },
        'auc_macro': {
            'mean': float(auc_mean), 
            'std': float(auc_std),
            'ci_95': [float(auc_mean - 1.96*auc_se), float(auc_mean + 1.96*auc_se)]
        }
    },
    'fold_scores': {
        'f1': [float(x) for x in fold_f1],
        'accuracy': [float(x) for x in fold_acc],
        'auc': [float(x) for x in fold_auc]
    },
    'per_class_f1': {LABEL_MAP[i]: float(f1_class[i]) for i in range(6)}
}

out_path = RESULTS_DIR / 'option6_full_metrics.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

print("\nDone!")
