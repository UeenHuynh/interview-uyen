#!/usr/bin/env python3
"""
Stage 2 Alternative: Direct Group PCA (No VIP/Stability)

Problem with previous approach:
- VIP + Stability Selection may remove class-specific signals
- LASSO optimizes for overall prediction, not per-class
- Gastric signal may have been destroyed

New approach:
- Skip VIP and Stability Selection entirely
- Apply PCA directly to each biological group (EM, FLEN, NUCLEOSOME)
- Keep top N PCs per group
- This preserves signal from each biological view

Rationale:
- PCA within each group captures that group's variance
- No single group can dominate/dilute another
- Gastric signal preserved if it exists in any group
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
METADATA_DIR = DATA_DIR / 'metadata'

# PCA settings
N_COMPONENTS_PER_GROUP = 5  # 5 PCs per group → 15 total
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# ============================================================
# Load Data
# ============================================================
print("=" * 70)
print("Stage 2 Alternative: Direct Group PCA")
print("=" * 70)

# Load scaled training data (after QC, before feature selection)
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_scaled.parquet')
X_test = pd.read_parquet(PROCESSED_DIR / 'X_test_scaled.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nLoaded data (after QC, before feature selection):")
print(f"  Train: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} samples x {X_test.shape[1]} features")

# ============================================================
# Identify Feature Groups
# ============================================================
print("\n" + "-" * 70)
print("Feature groups in QC'd data:")
print("-" * 70)

groups = {
    'EM': [c for c in X_train.columns if c.startswith('EM_')],
    'FLEN': [c for c in X_train.columns if c.startswith('FLEN_')],
    'NUCLEOSOME': [c for c in X_train.columns if c.startswith('NUC_')]
}

for name, features in groups.items():
    print(f"  {name}: {len(features)} features")

# ============================================================
# Apply Group PCA (Direct, No Pre-filtering)
# ============================================================
print("\n" + "-" * 70)
print(f"Applying Group PCA ({N_COMPONENTS_PER_GROUP} PCs per group)...")
print("-" * 70)

pca_models = {}
train_pca_dfs = []
test_pca_dfs = []

for group_name, features in groups.items():
    if len(features) == 0:
        print(f"  {group_name}: No features, skipping")
        continue

    # Determine number of components
    n_comp = min(N_COMPONENTS_PER_GROUP, len(features), X_train.shape[0] - 1)

    # Extract group features
    X_train_group = X_train[features].values
    X_test_group = X_test[features].values

    # Fit PCA on training data
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_group)
    X_test_pca = pca.transform(X_test_group)

    # Create column names
    col_names = [f"{group_name}_PC{i+1}" for i in range(n_comp)]

    # Create DataFrames
    train_df = pd.DataFrame(X_train_pca, columns=col_names, index=X_train.index)
    test_df = pd.DataFrame(X_test_pca, columns=col_names, index=X_test.index)

    train_pca_dfs.append(train_df)
    test_pca_dfs.append(test_df)

    # Save PCA model info
    pca_models[group_name] = {
        'model': pca,
        'n_features_input': len(features),
        'n_components': n_comp,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(pca.explained_variance_ratio_.sum()),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
    }

    # Print details
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"\n  {group_name}:")
    print(f"    Input features: {len(features)}")
    print(f"    Output PCs: {n_comp}")
    print(f"    Total variance explained: {var_explained:.1%}")
    print(f"    Per-PC variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        cum_var = np.sum(pca.explained_variance_ratio_[:i+1])
        bar = "█" * int(var * 50)
        print(f"      PC{i+1}: {var:.1%} (cumulative: {cum_var:.1%}) {bar}")

# Concatenate all groups
X_train_final = pd.concat(train_pca_dfs, axis=1)
X_test_final = pd.concat(test_pca_dfs, axis=1)

print(f"\n" + "-" * 70)
print(f"Final dimensions: {X_train_final.shape[1]} features")
print(f"Features: {list(X_train_final.columns)}")

# ============================================================
# Compare with Previous Approach
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: Previous vs New Approach")
print("=" * 70)

print("""
Previous (VIP + Stability + Group PCA):
  541 → 120 (VIP) → 38 (Stability) → 14 (Group PCA)
  Problem: Stability Selection may remove class-specific signals

New (Direct Group PCA):
  541 → 15 (Group PCA directly)
  Benefit: Each biological group contributes equally
           No signal loss from aggressive filtering
""")

# ============================================================
# Save Outputs
# ============================================================
print("-" * 70)
print("Saving outputs...")
print("-" * 70)

# Save with different filename to not overwrite
X_train_final.to_parquet(PROCESSED_DIR / 'X_train_final_v2.parquet', compression='snappy')
X_test_final.to_parquet(PROCESSED_DIR / 'X_test_final_v2.parquet', compression='snappy')
print(f"  Saved: X_train_final_v2.parquet, X_test_final_v2.parquet")

# Save PCA models
with open(METADATA_DIR / 'pca_models_v2.pkl', 'wb') as f:
    pickle.dump(pca_models, f)
print(f"  Saved: pca_models_v2.pkl")

# Save metadata
meta = {
    'approach': 'Direct Group PCA (no VIP/Stability)',
    'n_features_input': int(X_train.shape[1]),
    'n_features_output': int(X_train_final.shape[1]),
    'n_components_per_group': N_COMPONENTS_PER_GROUP,
    'groups': {
        name: {
            'n_input': info['n_features_input'],
            'n_output': info['n_components'],
            'variance_explained': info['total_variance_explained']
        }
        for name, info in pca_models.items()
    },
    'final_features': X_train_final.columns.tolist()
}

with open(METADATA_DIR / 'selected_features_v2.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  Saved: selected_features_v2.json")

# ============================================================
# Quick Baseline Test
# ============================================================
print("\n" + "=" * 70)
print("Quick Baseline Test (Logistic Regression)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}

# Compare old vs new features
print("\nComparing feature sets:")

# Load old features
X_train_old = pd.read_parquet(PROCESSED_DIR / 'X_train_final.parquet')
X_test_old = pd.read_parquet(PROCESSED_DIR / 'X_test_final.parquet')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=2000, random_state=42)

# Old features
scores_old = cross_val_score(model, X_train_old, y_train, cv=cv, scoring='f1_macro')
y_pred_old = cross_val_predict(model, X_train_old, y_train, cv=cv)
f1_per_class_old = f1_score(y_train, y_pred_old, average=None)

# New features
scores_new = cross_val_score(model, X_train_final, y_train, cv=cv, scoring='f1_macro')
y_pred_new = cross_val_predict(model, X_train_final, y_train, cv=cv)
f1_per_class_new = f1_score(y_train, y_pred_new, average=None)

print(f"\n{'Metric':<20} {'Old (VIP+Stab)':<15} {'New (Direct)':<15} {'Change':<10}")
print("-" * 60)
print(f"{'F1 (macro)':<20} {scores_old.mean():.3f} ± {scores_old.std():.3f}   {scores_new.mean():.3f} ± {scores_new.std():.3f}   {scores_new.mean() - scores_old.mean():+.3f}")

print(f"\nPer-class F1:")
print(f"{'Class':<12} {'Old':<10} {'New':<10} {'Change':<10}")
print("-" * 42)
for i, name in LABEL_MAP.items():
    old_f1 = f1_per_class_old[i]
    new_f1 = f1_per_class_new[i]
    change = new_f1 - old_f1
    indicator = "↑" if change > 0.05 else ("↓" if change < -0.05 else "→")
    highlight = "***" if name == "Gastric" else ""
    print(f"{name:<12} {old_f1:.3f}      {new_f1:.3f}      {change:+.3f} {indicator} {highlight}")

# Gastric specific
print(f"\n🎯 GASTRIC FOCUS:")
print(f"   Old F1: {f1_per_class_old[3]:.3f}")
print(f"   New F1: {f1_per_class_new[3]:.3f}")
print(f"   Change: {f1_per_class_new[3] - f1_per_class_old[3]:+.3f}")

if f1_per_class_new[3] > f1_per_class_old[3]:
    print(f"   ✓ Gastric improved with Direct Group PCA!")
else:
    print(f"   ⚠️ Gastric still poor - may be biological limitation")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Direct Group PCA approach:
  - EM:         {groups['EM'].__len__()} features → {pca_models['EM']['n_components']} PCs ({pca_models['EM']['total_variance_explained']:.1%} var)
  - FLEN:       {groups['FLEN'].__len__()} features → {pca_models['FLEN']['n_components']} PCs ({pca_models['FLEN']['total_variance_explained']:.1%} var)
  - NUCLEOSOME: {groups['NUCLEOSOME'].__len__()} features → {pca_models['NUCLEOSOME']['n_components']} PCs ({pca_models['NUCLEOSOME']['total_variance_explained']:.1%} var)

  Total: {X_train.shape[1]} → {X_train_final.shape[1]} features

Files saved:
  - X_train_final_v2.parquet
  - X_test_final_v2.parquet
  - pca_models_v2.pkl
  - selected_features_v2.json
""")
