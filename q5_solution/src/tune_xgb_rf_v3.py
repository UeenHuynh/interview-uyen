#!/usr/bin/env python3
"""Tune XGBoost and RandomForest for v3 features (5-fold CV)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load v3 features
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print("=" * 70)
print("Tuning XGBoost & RandomForest (v3 features)")
print("=" * 70)
print(f"Data: {X_train.shape[0]} samples x {X_train.shape[1]} features")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ------------------ Random Forest ------------------
rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)

rf_grid = {
    'n_estimators': [200, 400],
    'max_depth': [None, 6, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\n[RF] Grid search...")
rf_search = GridSearchCV(
    rf,
    rf_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    refit=True
)
rf_search.fit(X_train, y_train)

print(f"[RF] Best score: {rf_search.best_score_:.3f}")
print(f"[RF] Best params: {rf_search.best_params_}")

# ------------------ XGBoost ------------------
xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric='mlogloss',
    n_jobs=-1
)

xgb_grid = {
    'n_estimators': [200, 400],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

print("\n[XGB] Grid search...")
xgb_search = GridSearchCV(
    xgb,
    xgb_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    refit=True
)
xgb_search.fit(X_train, y_train)

print(f"[XGB] Best score: {xgb_search.best_score_:.3f}")
print(f"[XGB] Best params: {xgb_search.best_params_}")

# Save results
results = {
    'feature_set': 'v3',
    'rf': {
        'best_score': float(rf_search.best_score_),
        'best_params': rf_search.best_params_
    },
    'xgb': {
        'best_score': float(xgb_search.best_score_),
        'best_params': xgb_search.best_params_
    }
}

out_path = RESULTS_DIR / 'xgb_rf_tuning_v3.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {out_path}")
