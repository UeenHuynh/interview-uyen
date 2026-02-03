#!/usr/bin/env python3
"""
Stage 2: Scientific Feature Selection
- Layer A: Sparse PLS-DA with VIP scoring
- Layer B: Stability Selection (LASSO bootstrap)
- Layer C: Group-aware PCA for each feature group
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
METADATA_DIR = DATA_DIR / 'metadata'

# Feature selection parameters
N_BOOTSTRAP = 100       # Number of bootstrap iterations for stability selection
STABILITY_THRESHOLD = 0.50  # Minimum selection frequency (50% of iterations)
MAX_FEATURES_PLSDA = 150    # Target max features after PLS-DA
N_PCA_COMPONENTS_PER_GROUP = 5  # Max PCA components per group

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Stage 2: Scientific Feature Selection")
print("=" * 60)

X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_scaled.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values
X_test = pd.read_parquet(PROCESSED_DIR / 'X_test_scaled.parquet')

# Load feature groups info
with open(METADATA_DIR / 'feature_groups.json', 'r') as f:
    meta = json.load(f)

print(f"Loaded training data: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"Target: reduce to < {X_train.shape[0] // 2} features (n/2 rule)")

# ============================================================
# Layer A: Sparse PLS-DA with VIP Scoring
# ============================================================
print("\n" + "-" * 40)
print("Layer A: Sparse PLS-DA with VIP scoring...")

# Determine number of components (rule: < n/3)
n_components = min(10, len(X_train) // 3)
print(f"  Using {n_components} PLS components")

# Fit PLS model
pls = PLSRegression(n_components=n_components, scale=False)  # Already scaled
pls.fit(X_train.values, y_train)

# Calculate VIP (Variable Importance in Projection) scores
def calculate_vip(pls_model, X):
    """
    Calculate VIP scores for PLS model.
    VIP = sqrt(p * sum(W^2 * SSY) / sum(SSY))
    """
    W = pls_model.x_weights_       # (p, n_components)
    T = pls_model.x_scores_        # (n, n_components)
    Q = pls_model.y_loadings_      # (1, n_components)

    p = X.shape[1]
    n_comp = W.shape[1]

    # Sum of squares of Y explained by each component
    SSY = np.sum(T**2, axis=0) * Q.flatten()**2

    # VIP calculation
    VIP = np.sqrt(p * np.sum((W**2) * (SSY / SSY.sum()), axis=1))
    return VIP

vip_scores = calculate_vip(pls, X_train.values)

# Create VIP dataframe
vip_df = pd.DataFrame({
    'feature': X_train.columns,
    'vip_score': vip_scores
}).sort_values('vip_score', ascending=False)

# Select top features by VIP (target: ~150 features)
n_select_vip = min(MAX_FEATURES_PLSDA, len(X_train) // 2)
top_vip_features = vip_df.head(n_select_vip)['feature'].tolist()

print(f"  VIP scores range: [{vip_scores.min():.4f}, {vip_scores.max():.4f}]")
print(f"  Selected top {n_select_vip} features by VIP score")
print(f"  VIP threshold (min selected): {vip_df.head(n_select_vip)['vip_score'].min():.4f}")

# Breakdown by feature group
for prefix, name in [('EM_', 'EM'), ('FLEN_', 'FLEN'), ('NUC_', 'NUCLEOSOME')]:
    count = len([f for f in top_vip_features if f.startswith(prefix)])
    print(f"    {name}: {count} features selected")

# ============================================================
# Layer B: Stability Selection
# ============================================================
print("\n" + "-" * 40)
print("Layer B: Stability Selection (LASSO bootstrap)...")
print(f"  Running {N_BOOTSTRAP} bootstrap iterations...")

# Subset to VIP-selected features
X_train_vip = X_train[top_vip_features].values

# Track feature selection frequency
feature_counts = np.zeros(len(top_vip_features))

for i in range(N_BOOTSTRAP):
    if (i + 1) % 20 == 0:
        print(f"    Iteration {i+1}/{N_BOOTSTRAP}")

    # Bootstrap subsample (50% of training data)
    n_subsample = len(X_train_vip) // 2
    idx = resample(range(len(X_train_vip)), n_samples=n_subsample, replace=False, random_state=i)

    X_sub = X_train_vip[idx]
    y_sub = y_train[idx]

    # Fit LASSO with cross-validation
    try:
        lasso = LassoCV(cv=5, random_state=i, max_iter=5000)
        lasso.fit(X_sub, y_sub)

        # Count non-zero coefficients
        selected = np.abs(lasso.coef_) > 1e-5
        feature_counts[selected] += 1
    except Exception as e:
        continue

# Calculate selection frequency
selection_freq = feature_counts / N_BOOTSTRAP

# Select stable features (frequency >= threshold)
stable_mask = selection_freq >= STABILITY_THRESHOLD
stable_features = [top_vip_features[i] for i in range(len(top_vip_features)) if stable_mask[i]]

print(f"\n  Selection frequency range: [{selection_freq.min():.2f}, {selection_freq.max():.2f}]")
print(f"  Stability threshold: {STABILITY_THRESHOLD}")
print(f"  Stable features selected: {len(stable_features)}")

# If too few features, lower threshold
if len(stable_features) < 30:
    print(f"  Warning: Only {len(stable_features)} features. Lowering threshold...")
    STABILITY_THRESHOLD = 0.30
    stable_mask = selection_freq >= STABILITY_THRESHOLD
    stable_features = [top_vip_features[i] for i in range(len(top_vip_features)) if stable_mask[i]]
    print(f"  With threshold {STABILITY_THRESHOLD}: {len(stable_features)} features")

# Breakdown by feature group
for prefix, name in [('EM_', 'EM'), ('FLEN_', 'FLEN'), ('NUC_', 'NUCLEOSOME')]:
    count = len([f for f in stable_features if f.startswith(prefix)])
    print(f"    {name}: {count} stable features")

# ============================================================
# Layer C: Group-Aware PCA
# ============================================================
print("\n" + "-" * 40)
print("Layer C: Group-Aware PCA...")

# Get remaining features per group
groups = {
    'EM': [f for f in stable_features if f.startswith('EM_')],
    'FLEN': [f for f in stable_features if f.startswith('FLEN_')],
    'NUCLEOSOME': [f for f in stable_features if f.startswith('NUC_')]
}

# Apply PCA to each group separately
pca_models = {}
final_train_dfs = []
final_test_dfs = []

for group_name, features in groups.items():
    if len(features) == 0:
        print(f"  {group_name}: No features remaining, skipping")
        continue

    # Determine number of components
    n_comp = min(N_PCA_COMPONENTS_PER_GROUP, len(features), len(X_train) - 1)

    # Fit PCA on training data
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train[features].values)
    X_test_pca = pca.transform(X_test[features].values)

    # Create column names
    col_names = [f"{group_name}_PC{i+1}" for i in range(n_comp)]

    # Create DataFrames
    train_df = pd.DataFrame(X_train_pca, columns=col_names, index=X_train.index)
    test_df = pd.DataFrame(X_test_pca, columns=col_names, index=X_test.index)

    final_train_dfs.append(train_df)
    final_test_dfs.append(test_df)

    # Save PCA model
    pca_models[group_name] = {
        'model': pca,
        'features': features,
        'n_components': n_comp,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(pca.explained_variance_ratio_.sum())
    }

    print(f"  {group_name}: {len(features)} features -> {n_comp} PCs "
          f"({pca.explained_variance_ratio_.sum():.1%} variance)")

# Concatenate all groups
X_train_final = pd.concat(final_train_dfs, axis=1)
X_test_final = pd.concat(final_test_dfs, axis=1)

print(f"\n  Final dimensions: {X_train_final.shape[1]} features (train: {len(X_train_final)}, test: {len(X_test_final)})")

# ============================================================
# Save Outputs
# ============================================================
print("\n" + "-" * 40)
print("Saving outputs...")

# Save final features
X_train_final.to_parquet(PROCESSED_DIR / 'X_train_final.parquet', compression='snappy')
X_test_final.to_parquet(PROCESSED_DIR / 'X_test_final.parquet', compression='snappy')
print(f"  Saved: X_train_final.parquet, X_test_final.parquet")

# Save VIP scores
vip_df.to_csv(PROCESSED_DIR / 'vip_scores.csv', index=False)
print(f"  Saved: vip_scores.csv")

# Save stability selection results
stability_df = pd.DataFrame({
    'feature': top_vip_features,
    'selection_frequency': selection_freq,
    'stable': stable_mask
})
stability_df.to_csv(PROCESSED_DIR / 'stability_selection.csv', index=False)
print(f"  Saved: stability_selection.csv")

# Save PCA models
with open(METADATA_DIR / 'pca_models.pkl', 'wb') as f:
    pickle.dump(pca_models, f)
print(f"  Saved: pca_models.pkl")

# Save selected features list
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

with open(METADATA_DIR / 'selected_features.json', 'w') as f:
    json.dump(selected_features_meta, f, indent=2)
print(f"  Saved: selected_features.json")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Stage 2 Complete!")
print("=" * 60)
print(f"Feature reduction pipeline:")
print(f"  Original (after QC):  {X_train.shape[1]} features")
print(f"  After PLS-DA VIP:     {len(top_vip_features)} features")
print(f"  After Stability:      {len(stable_features)} features")
print(f"  After Group PCA:      {X_train_final.shape[1]} features")
print(f"\nReduction ratio: {X_train.shape[1]} -> {X_train_final.shape[1]} ({100*(1-X_train_final.shape[1]/X_train.shape[1]):.1f}% removed)")
print(f"\nFinal feature breakdown:")
for group_name, info in pca_models.items():
    print(f"  {group_name}: {info['n_components']} PCs ({info['total_variance_explained']:.1%} variance)")
