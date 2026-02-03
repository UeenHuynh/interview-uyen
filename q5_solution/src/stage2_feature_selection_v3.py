#!/usr/bin/env python3
"""
Stage 2 (v3): Feature Engineering on Raw Features
- Layer A: Sparse PLS-DA with VIP scoring -> 50 features
- Layer B: Stability Selection (L1 logistic bootstrap) -> 30 features
- Layer C: Group-aware PCA -> 15 dims total
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
METADATA_DIR = DATA_DIR / 'metadata'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Layer A
N_PLS_COMPONENTS = 5
MAX_FEATURES_PLSDA = 50

# Layer B
N_BOOTSTRAP = 100
SUBSAMPLE_FRACTION = 0.8
STABILITY_TARGET = 30
STABILITY_THRESHOLD = 0.60

# Layer C
GROUP_PCA_TOTAL_COMPONENTS = 15

# ============================================================
# Helpers
# ============================================================

def calculate_vip(pls_model, X, y_scores):
    """
    Calculate VIP scores for a multi-response PLS model.
    VIP = sqrt(p * sum_k( w_jk^2 * SSY_k / sum(SSY) ))
    """
    W = pls_model.x_weights_  # (p, n_components)
    T = pls_model.x_scores_   # (n, n_components)

    # Sum of squares of Y explained by each component
    # y_scores is (n, n_targets)
    # Q is (n_targets, n_components)
    Q = pls_model.y_loadings_
    ssy = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)

    p = X.shape[1]
    vip = np.sqrt(p * np.sum((W ** 2) * (ssy / ssy.sum()), axis=1))
    return vip


def allocate_group_components(group_sizes, total_components):
    """
    Allocate PCA components across groups proportionally,
    ensuring each non-empty group gets at least 1 component.
    """
    groups = [g for g, n in group_sizes.items() if n > 0]
    if not groups:
        return {g: 0 for g in group_sizes}

    # Start with proportional allocation
    total_features = sum(group_sizes[g] for g in groups)
    raw_alloc = {g: (group_sizes[g] / total_features) * total_components for g in groups}
    alloc = {g: max(1, int(np.floor(raw_alloc[g]))) for g in groups}

    # Adjust to match total_components
    current_total = sum(alloc.values())
    remainder = total_components - current_total

    if remainder > 0:
        # Distribute remaining components by largest fractional parts
        frac_order = sorted(groups, key=lambda g: raw_alloc[g] - np.floor(raw_alloc[g]), reverse=True)
        for g in frac_order:
            if remainder == 0:
                break
            alloc[g] += 1
            remainder -= 1
    elif remainder < 0:
        # Remove components from largest allocations (but keep at least 1)
        over_order = sorted(groups, key=lambda g: alloc[g], reverse=True)
        for g in over_order:
            if remainder == 0:
                break
            if alloc[g] > 1:
                alloc[g] -= 1
                remainder += 1

    # Fill missing groups with 0 components
    for g in group_sizes:
        if g not in alloc:
            alloc[g] = 0

    return alloc


# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Stage 2 (v3): Feature Engineering on Raw Features")
print("=" * 60)

X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_scaled.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values
X_test = pd.read_parquet(PROCESSED_DIR / 'X_test_scaled.parquet')

with open(METADATA_DIR / 'feature_groups.json', 'r') as f:
    meta = json.load(f)

print(f"Loaded training data: {X_train.shape[0]} samples x {X_train.shape[1]} features")

# ============================================================
# Layer A: Sparse PLS-DA with VIP Scoring
# ============================================================
print("\n" + "-" * 40)
print("Layer A: Sparse PLS-DA with VIP scoring...")

# One-hot encode labels for multi-class PLS-DA
enc = OneHotEncoder(sparse_output=False)
y_onehot = enc.fit_transform(y_train.reshape(-1, 1))

n_components = min(N_PLS_COMPONENTS, len(X_train) - 1)
print(f"  Using {n_components} PLS components")

pls = PLSRegression(n_components=n_components, scale=False)
pls.fit(X_train.values, y_onehot)

vip_scores = calculate_vip(pls, X_train.values, y_onehot)

vip_df = pd.DataFrame({
    'feature': X_train.columns,
    'vip_score': vip_scores
}).sort_values('vip_score', ascending=False)

n_select_vip = min(MAX_FEATURES_PLSDA, X_train.shape[1])
top_vip_features = vip_df.head(n_select_vip)['feature'].tolist()

print(f"  VIP scores range: [{vip_scores.min():.4f}, {vip_scores.max():.4f}]")
print(f"  Selected top {n_select_vip} features by VIP score")
print(f"  VIP threshold (min selected): {vip_df.head(n_select_vip)['vip_score'].min():.4f}")

# ============================================================
# Layer B: Stability Selection (L1 Logistic)
# ============================================================
print("\n" + "-" * 40)
print("Layer B: Stability Selection (L1 Logistic bootstrap)...")
print(f"  Running {N_BOOTSTRAP} bootstrap iterations...")

X_train_vip = X_train[top_vip_features].values

feature_counts = np.zeros(len(top_vip_features))

for i in range(N_BOOTSTRAP):
    if (i + 1) % 20 == 0:
        print(f"    Iteration {i+1}/{N_BOOTSTRAP}")

    n_subsample = int(len(X_train_vip) * SUBSAMPLE_FRACTION)
    idx = resample(range(len(X_train_vip)), n_samples=n_subsample, replace=False, random_state=i)

    X_sub = X_train_vip[idx]
    y_sub = y_train[idx]

    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        multi_class='multinomial',
        max_iter=2000,
        random_state=i,
    )
    try:
        model.fit(X_sub, y_sub)
        coefs = model.coef_  # (n_classes, n_features)
        selected = np.any(np.abs(coefs) > 1e-6, axis=0)
        feature_counts[selected] += 1
    except Exception:
        continue

selection_freq = feature_counts / N_BOOTSTRAP
stable_mask = selection_freq >= STABILITY_THRESHOLD
stable_features = [top_vip_features[i] for i in range(len(top_vip_features)) if stable_mask[i]]

print(f"\n  Selection frequency range: [{selection_freq.min():.2f}, {selection_freq.max():.2f}]")
print(f"  Stability threshold: {STABILITY_THRESHOLD}")
print(f"  Stable features selected: {len(stable_features)}")

# If too many/too few, enforce target by top frequency
if len(stable_features) != STABILITY_TARGET:
    top_idx = np.argsort(selection_freq)[-STABILITY_TARGET:]
    stable_features = [top_vip_features[i] for i in top_idx]
    stable_mask = np.zeros_like(selection_freq, dtype=bool)
    stable_mask[top_idx] = True
    print(f"  Adjusted to top {STABILITY_TARGET} features by stability score")

# ============================================================
# Layer C: Group PCA (total 15 dims)
# ============================================================
print("\n" + "-" * 40)
print("Layer C: Group-aware PCA...")

# Map selected features to groups by prefix
groups = {
    'EM': [f for f in stable_features if f.startswith('EM_')],
    'FLEN': [f for f in stable_features if f.startswith('FLEN_')],
    'NUCLEOSOME': [f for f in stable_features if f.startswith('NUC_')]
}

alloc = allocate_group_components(
    {g: len(f) for g, f in groups.items()},
    GROUP_PCA_TOTAL_COMPONENTS,
)

pca_models = {}
final_train_dfs = []
final_test_dfs = []

for group_name, features in groups.items():
    if len(features) == 0 or alloc[group_name] == 0:
        print(f"  {group_name}: No features remaining, skipping")
        continue

    n_comp = min(alloc[group_name], len(features), len(X_train) - 1)

    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train[features].values)
    X_test_pca = pca.transform(X_test[features].values)

    col_names = [f"{group_name}_PC{i+1}" for i in range(n_comp)]

    train_df = pd.DataFrame(X_train_pca, columns=col_names, index=X_train.index)
    test_df = pd.DataFrame(X_test_pca, columns=col_names, index=X_test.index)

    final_train_dfs.append(train_df)
    final_test_dfs.append(test_df)

    pca_models[group_name] = {
        'model': pca,
        'features': features,
        'n_components': n_comp,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(pca.explained_variance_ratio_.sum())
    }

    print(f"  {group_name}: {len(features)} features -> {n_comp} PCs "
          f"({pca.explained_variance_ratio_.sum():.1%} variance)")

X_train_final = pd.concat(final_train_dfs, axis=1)
X_test_final = pd.concat(final_test_dfs, axis=1)

print(f"\n  Final dimensions: {X_train_final.shape[1]} features (train: {len(X_train_final)}, test: {len(X_test_final)})")

# ============================================================
# Save Outputs (v3)
# ============================================================
print("\n" + "-" * 40)
print("Saving outputs (v3)...")

X_train_final.to_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet', compression='snappy')
X_test_final.to_parquet(PROCESSED_DIR / 'X_test_final_v3.parquet', compression='snappy')
print("  Saved: X_train_final_v3.parquet, X_test_final_v3.parquet")

vip_df.to_csv(PROCESSED_DIR / 'vip_scores_v3.csv', index=False)
print("  Saved: vip_scores_v3.csv")

stability_df = pd.DataFrame({
    'feature': top_vip_features,
    'selection_frequency': selection_freq,
    'stable': stable_mask
})
stability_df.to_csv(PROCESSED_DIR / 'stability_selection_v3.csv', index=False)
print("  Saved: stability_selection_v3.csv")

with open(METADATA_DIR / 'pca_models_v3.pkl', 'wb') as f:
    pickle.dump(pca_models, f)
print("  Saved: pca_models_v3.pkl")

selected_features_meta = {
    'vip_selected': top_vip_features,
    'stable_selected': stable_features,
    'final_pca_features': X_train_final.columns.tolist(),
    'n_features_original': int(X_train.shape[1]),
    'n_features_vip': len(top_vip_features),
    'n_features_stable': len(stable_features),
    'n_features_final': int(X_train_final.shape[1]),
    'pca_variance_explained': {
        group: info['total_variance_explained']
        for group, info in pca_models.items()
    }
}

with open(METADATA_DIR / 'selected_features_v3.json', 'w') as f:
    json.dump(selected_features_meta, f, indent=2)
print("  Saved: selected_features_v3.json")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Stage 2 (v3) Complete!")
print("=" * 60)
print("Feature reduction pipeline:")
print(f"  Original (after QC):  {X_train.shape[1]} features")
print(f"  After PLS-DA VIP:     {len(top_vip_features)} features")
print(f"  After Stability:      {len(stable_features)} features")
print(f"  After Group PCA:      {X_train_final.shape[1]} features")
print(f"\nReduction ratio: {X_train.shape[1]} -> {X_train_final.shape[1]} "
      f"({100*(1 - X_train_final.shape[1]/X_train.shape[1]):.1f}% removed)")
for group_name, info in pca_models.items():
    print(f"  {group_name}: {info['n_components']} PCs ({info['total_variance_explained']:.1%} variance)")
