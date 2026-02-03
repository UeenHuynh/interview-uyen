#!/usr/bin/env python3
"""Voting-only evaluation for v3 features (5-fold CV)."""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}

FEATURE_SET = os.environ.get('FEATURE_SET', 'v3').lower()
FEATURE_SUFFIX = '_v3' if FEATURE_SET == 'v3' else ''
USE_TUNED_LR_SVM = os.environ.get('USE_TUNED_LR_SVM', '1') == '1'
USE_TUNED_XGB_RF = os.environ.get('USE_TUNED_XGB_RF', '1') == '1'
OUTPUT_TAG = os.environ.get('OUTPUT_TAG', '').strip()

# Load data
X_train = pd.read_parquet(PROCESSED_DIR / f'X_train_final{FEATURE_SUFFIX}.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

# Load tuned params for LR/SVM if available
phase2_tuned_path = RESULTS_DIR / f'phase2_tuning_results{FEATURE_SUFFIX}.json'
xgb_rf_tuned_path = RESULTS_DIR / 'xgb_rf_tuning_v3.json'

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
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
default_lr_params = {}
default_svm_params = {'C': 1.0, 'gamma': 'scale'}

if USE_TUNED_LR_SVM and phase2_tuned_path.exists():
    with open(phase2_tuned_path) as f:
        phase2_results = json.load(f)
    lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', default_lr_params)
    svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', default_svm_params)
    print(f"Using Phase 2 tuned params from {phase2_tuned_path.name}")
else:
    lr_params = default_lr_params
    svm_params = default_svm_params
    print("Phase 2 tuned params not found. Using defaults for LR/SVM")

if USE_TUNED_XGB_RF and xgb_rf_tuned_path.exists():
    with open(xgb_rf_tuned_path) as f:
        xgb_rf_results = json.load(f)
    xgb_params = xgb_rf_results.get('xgb', {}).get('best_params', default_xgb_params)
    rf_params = xgb_rf_results.get('rf', {}).get('best_params', default_rf_params)
    print(f"Using tuned XGB/RF params from {xgb_rf_tuned_path.name}")
else:
    xgb_params = default_xgb_params
    rf_params = default_rf_params
    print("XGB/RF tuned params not found. Using defaults for XGB/RF")

# Voting CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

voting_fold_scores = []
voting_all_preds = np.zeros(len(y_train), dtype=int)

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\n--- Fold {fold_idx + 1}/5 ---")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    sample_weights = compute_sample_weight('balanced', y_tr)

    lr_kwargs = dict(lr_params)
    lr_kwargs.setdefault('class_weight', 'balanced')
    lr_model = LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE)

    svm_kwargs = dict(svm_params)
    svm_kwargs.setdefault('class_weight', 'balanced')
    svm_model = SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True)

    rf_model = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    xgb_model = XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)

    lr_model.fit(X_tr, y_tr)
    svm_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)

    proba_lr = lr_model.predict_proba(X_val)
    proba_svm = svm_model.predict_proba(X_val)
    proba_rf = rf_model.predict_proba(X_val)
    proba_xgb = xgb_model.predict_proba(X_val)

    avg_proba = (proba_lr + proba_svm + proba_rf + proba_xgb) / 4.0
    voting_pred = np.argmax(avg_proba, axis=1)

    f1 = f1_score(y_val, voting_pred, average='macro')
    voting_fold_scores.append(f1)
    voting_all_preds[val_idx] = voting_pred

    print(f"  Voting: {f1:.3f}")

voting_mean = np.mean(voting_fold_scores)
voting_std = np.std(voting_fold_scores)

voting_f1_class = f1_score(y_train, voting_all_preds, average=None)
cm = confusion_matrix(y_train, voting_all_preds)

print("\nRESULTS SUMMARY")
print(f"Voting F1 macro: {voting_mean:.3f} ± {voting_std:.3f}")
print("Per-class F1:")
for i, name in LABEL_MAP.items():
    print(f"  {name}: {voting_f1_class[i]:.3f}")

# Save
results = {
    'feature_set': FEATURE_SET,
    'cv_folds': 5,
    'voting': {
        'f1_macro_mean': float(voting_mean),
        'f1_macro_std': float(voting_std),
        'per_class': {LABEL_MAP[i]: float(voting_f1_class[i]) for i in range(6)},
        'fold_scores': [float(s) for s in voting_fold_scores],
        'confusion_matrix': cm.tolist()
    }
}

tag_part = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
out_path = RESULTS_DIR / f'voting_results_{FEATURE_SET}{tag_part}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# Save fitted voting models on full data
lr_kwargs = dict(lr_params)
lr_kwargs.setdefault('class_weight', 'balanced')
lr_full = LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE)

svm_kwargs = dict(svm_params)
svm_kwargs.setdefault('class_weight', 'balanced')
svm_full = SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True)

rf_full = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
xgb_full = XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)

sample_weights_full = compute_sample_weight('balanced', y_train)

lr_full.fit(X_train, y_train)
svm_full.fit(X_train, y_train)
rf_full.fit(X_train, y_train)
xgb_full.fit(X_train, y_train, sample_weight=sample_weights_full)

models = {
    'feature_set': FEATURE_SET,
    'lr': lr_full,
    'svm': svm_full,
    'rf': rf_full,
    'xgb': xgb_full
}

model_path = RESULTS_DIR / f'voting_models_{FEATURE_SET}{tag_part}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(models, f)
print(f"Saved: {model_path}")
