#!/usr/bin/env python3
"""Compute AUC and Accuracy for final voting model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, 
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}
RANDOM_STATE = 42

# Load data
print("Loading data...")
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

# Load tuned params
phase2_tuned_path = RESULTS_DIR / 'phase2_tuning_results_v3.json'
xgb_rf_tuned_path = RESULTS_DIR / 'xgb_rf_tuning_v3.json'

if phase2_tuned_path.exists():
    with open(phase2_tuned_path) as f:
        phase2_results = json.load(f)
    lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', {})
    svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', {'C': 1.0, 'gamma': 'scale'})
else:
    lr_params = {}
    svm_params = {'C': 1.0, 'gamma': 'scale'}

if xgb_rf_tuned_path.exists():
    with open(xgb_rf_tuned_path) as f:
        xgb_rf_results = json.load(f)
    xgb_params = xgb_rf_results.get('xgb', {}).get('best_params', {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1})
    rf_params = xgb_rf_results.get('rf', {}).get('best_params', {'n_estimators': 200})
else:
    xgb_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
    rf_params = {'n_estimators': 200}

# 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_f1 = []
fold_acc = []
fold_auc = []

all_y_true = []
all_y_pred = []
all_y_proba = []

print("\nRunning 5-Fold CV with Voting Ensemble...\n")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"--- Fold {fold_idx + 1}/5 ---")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    sample_weights = compute_sample_weight('balanced', y_tr)
    
    # Build models
    lr_kwargs = dict(lr_params)
    lr_kwargs.setdefault('class_weight', 'balanced')
    lr_model = LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE)
    
    svm_kwargs = dict(svm_params)
    svm_kwargs.setdefault('class_weight', 'balanced')
    svm_model = SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True)
    
    rf_model = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    xgb_model = XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
    
    # Fit
    lr_model.fit(X_tr, y_tr)
    svm_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)
    
    # Predict probabilities
    proba_lr = lr_model.predict_proba(X_val)
    proba_svm = svm_model.predict_proba(X_val)
    proba_rf = rf_model.predict_proba(X_val)
    proba_xgb = xgb_model.predict_proba(X_val)
    
    # Soft voting
    avg_proba = (proba_lr + proba_svm + proba_rf + proba_xgb) / 4.0
    voting_pred = np.argmax(avg_proba, axis=1)
    
    # Metrics
    f1 = f1_score(y_val, voting_pred, average='macro')
    acc = accuracy_score(y_val, voting_pred)
    
    # AUC (one-vs-rest for multi-class)
    y_val_bin = label_binarize(y_val, classes=[0, 1, 2, 3, 4, 5])
    auc = roc_auc_score(y_val_bin, avg_proba, average='macro', multi_class='ovr')
    
    fold_f1.append(f1)
    fold_acc.append(acc)
    fold_auc.append(auc)
    
    all_y_true.extend(y_val)
    all_y_pred.extend(voting_pred)
    all_y_proba.extend(avg_proba)
    
    print(f"  F1 Macro: {f1:.3f} | Accuracy: {acc:.3f} | AUC (macro): {auc:.3f}")

# Overall metrics
print("\n" + "="*60)
print("FINAL RESULTS (5-Fold CV)")
print("="*60)

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
    'cv_folds': 5,
    'metrics': {
        'f1_macro': {'mean': float(f1_mean), 'std': float(f1_std), 
                     'ci_95': [float(f1_mean - 1.96*f1_se), float(f1_mean + 1.96*f1_se)]},
        'accuracy': {'mean': float(acc_mean), 'std': float(acc_std),
                     'ci_95': [float(acc_mean - 1.96*acc_se), float(acc_mean + 1.96*acc_se)]},
        'auc_macro': {'mean': float(auc_mean), 'std': float(auc_std),
                      'ci_95': [float(auc_mean - 1.96*auc_se), float(auc_mean + 1.96*auc_se)]}
    },
    'fold_scores': {
        'f1': [float(x) for x in fold_f1],
        'accuracy': [float(x) for x in fold_acc],
        'auc': [float(x) for x in fold_auc]
    },
    'per_class_f1': {LABEL_MAP[i]: float(f1_class[i]) for i in range(6)}
}

with open(RESULTS_DIR / 'final_metrics_with_auc.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {RESULTS_DIR / 'final_metrics_with_auc.json'}")

print("\nDone!")
