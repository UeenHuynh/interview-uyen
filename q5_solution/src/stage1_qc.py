#!/usr/bin/env python3
"""
Stage 1: Quality Control & Preprocessing
- Step 1.1: Train/Test Split FIRST (stratified)
- Step 1.2: Remove zero-variance features (based on TRAIN only)
- Step 1.3: Correlation filtering (based on TRAIN only)
- Step 1.4: StandardScaler (fit on TRAIN only)

IMPORTANT: All filtering decisions are made on TRAINING data only to prevent data leakage.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
METADATA_DIR = DATA_DIR / 'metadata'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.90

# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Stage 1: Quality Control & Preprocessing")
print("=" * 60)

X = pd.read_parquet(PROCESSED_DIR / 'cleaned_data.parquet')
y = pd.read_parquet(PROCESSED_DIR / 'labels.parquet')['label']

print(f"Loaded data: {X.shape[0]} samples x {X.shape[1]} features")

# ============================================================
# Step 1.1: Train/Test Split FIRST (to prevent data leakage)
# ============================================================
print("\n" + "-" * 40)
print("Step 1.1: Stratified train/test split FIRST...")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"  Train set: {X_train_raw.shape[0]} samples x {X_train_raw.shape[1]} features")
print(f"  Test set:  {X_test_raw.shape[0]} samples x {X_test_raw.shape[1]} features")

# Check stratification
print(f"\n  Train label distribution:")
for label, name in [(0, 'Control'), (1, 'Breast'), (2, 'CRC'),
                     (3, 'Gastric'), (4, 'Liver'), (5, 'Lung')]:
    count = (y_train == label).sum()
    print(f"    {name}: {count} ({count/len(y_train)*100:.1f}%)")

# ============================================================
# Step 1.2: Remove Zero-Variance Features (based on TRAIN only)
# ============================================================
print("\n" + "-" * 40)
print("Step 1.2: Removing zero-variance features (based on TRAIN only)...")

# Find zero-variance features in TRAINING data only
train_variance = X_train_raw.var()
zero_var_mask = train_variance == 0
n_zero_var = zero_var_mask.sum()
zero_var_features = list(X_train_raw.columns[zero_var_mask])

print(f"  Found {n_zero_var} zero-variance features in training data")
if n_zero_var > 0:
    print(f"  Zero-variance features: {zero_var_features[:10]}...")

# Remove from both train and test using training mask
non_zero_var_features = X_train_raw.columns[~zero_var_mask].tolist()
X_train_qc = X_train_raw[non_zero_var_features]
X_test_qc = X_test_raw[non_zero_var_features]

print(f"  After removal: {X_train_qc.shape[1]} features remain")

# Track per-group removal
for prefix, name in [('EM_', 'EM'), ('FLEN_', 'FLEN'), ('NUC_', 'NUCLEOSOME')]:
    original = len([c for c in X_train_raw.columns if c.startswith(prefix)])
    remaining = len([c for c in X_train_qc.columns if c.startswith(prefix)])
    removed = original - remaining
    print(f"    {name}: {original} -> {remaining} (removed {removed})")

# ============================================================
# Step 1.3: Correlation Filtering (based on TRAIN only)
# ============================================================
print("\n" + "-" * 40)
print(f"Step 1.3: Correlation filtering based on TRAIN only (threshold={CORRELATION_THRESHOLD})...")

def remove_correlated_features_train_only(X_train, X_test, threshold=0.90):
    """
    Remove highly correlated features based on TRAINING data correlation.
    Apply the same removal to test data.
    """
    # Calculate correlation matrix on TRAINING data only
    corr_matrix = X_train.corr().abs()

    # Get upper triangle (avoid duplicate pairs)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation above threshold
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        if high_corr:
            # For each highly correlated pair, drop the one with lower variance in TRAIN
            for other in high_corr:
                if X_train[col].var() >= X_train[other].var():
                    to_drop.add(other)
                else:
                    to_drop.add(col)

    # Apply same removal to both train and test
    keep_features = [c for c in X_train.columns if c not in to_drop]
    return X_train[keep_features], X_test[keep_features], list(to_drop)

# Apply correlation filtering
X_train_uncorr, X_test_uncorr, dropped_corr = remove_correlated_features_train_only(
    X_train_qc, X_test_qc, threshold=CORRELATION_THRESHOLD
)

print(f"  Removed {len(dropped_corr)} highly correlated features")
print(f"  After removal: {X_train_uncorr.shape[1]} features remain")

# Track per-group removal
for prefix, name in [('EM_', 'EM'), ('FLEN_', 'FLEN'), ('NUC_', 'NUCLEOSOME')]:
    before = len([c for c in X_train_qc.columns if c.startswith(prefix)])
    after = len([c for c in X_train_uncorr.columns if c.startswith(prefix)])
    removed = before - after
    print(f"    {name}: {before} -> {after} (removed {removed})")

# ============================================================
# Step 1.4: StandardScaler (fit on TRAIN only)
# ============================================================
print("\n" + "-" * 40)
print("Step 1.4: StandardScaler (fit on TRAIN only)...")

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_uncorr)
X_test_scaled = scaler.transform(X_test_uncorr)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    index=X_train_uncorr.index,
    columns=X_train_uncorr.columns
)
X_test_scaled = pd.DataFrame(
    X_test_scaled,
    index=X_test_uncorr.index,
    columns=X_test_uncorr.columns
)

print(f"  Scaling complete (StandardScaler fitted on training data only)")
print(f"  Train mean: {X_train_scaled.mean().mean():.6f} (should be ~0)")
print(f"  Train std:  {X_train_scaled.std().mean():.6f} (should be ~1)")

# ============================================================
# Save Outputs
# ============================================================
print("\n" + "-" * 40)
print("Saving outputs...")

# Save QC'd data (before scaling, after QC)
X_train_uncorr.to_parquet(PROCESSED_DIR / 'qc_cleaned.parquet', compression='snappy')
print(f"  Saved: qc_cleaned.parquet")

# Save scaled training data
X_train_scaled.to_parquet(PROCESSED_DIR / 'X_train_scaled.parquet', compression='snappy')
print(f"  Saved: X_train_scaled.parquet")

# Save scaled test data
X_test_scaled.to_parquet(PROCESSED_DIR / 'X_test_scaled.parquet', compression='snappy')
print(f"  Saved: X_test_scaled.parquet")

# Save labels
y_train.to_frame().to_parquet(PROCESSED_DIR / 'y_train.parquet', compression='snappy')
y_test.to_frame().to_parquet(PROCESSED_DIR / 'y_test.parquet', compression='snappy')
print(f"  Saved: y_train.parquet, y_test.parquet")

# Save scaler
with open(METADATA_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Saved: scaler.pkl")

# Save QC report
qc_report = {
    "original_features": int(X.shape[1]),
    "zero_variance_removed": int(n_zero_var),
    "zero_variance_features": zero_var_features,
    "correlation_threshold": CORRELATION_THRESHOLD,
    "correlation_removed": len(dropped_corr),
    "final_features": int(X_train_uncorr.shape[1]),
    "features_remaining": X_train_uncorr.columns.tolist(),
    "train_samples": int(len(X_train_scaled)),
    "test_samples": int(len(X_test_scaled)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "data_leakage_prevention": "All QC decisions made on TRAIN data only",
    "feature_breakdown": {
        "EM": len([c for c in X_train_uncorr.columns if c.startswith('EM_')]),
        "FLEN": len([c for c in X_train_uncorr.columns if c.startswith('FLEN_')]),
        "NUCLEOSOME": len([c for c in X_train_uncorr.columns if c.startswith('NUC_')])
    }
}

with open(METADATA_DIR / 'qc_report.json', 'w') as f:
    json.dump(qc_report, f, indent=2)
print(f"  Saved: qc_report.json")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Stage 1 Complete!")
print("=" * 60)
print(f"Feature reduction: {X.shape[1]} -> {X_train_uncorr.shape[1]} ({100*(1-X_train_uncorr.shape[1]/X.shape[1]):.1f}% removed)")
print(f"  - Zero-variance removed: {n_zero_var}")
print(f"  - Correlation removed:   {len(dropped_corr)}")
print(f"\nFinal feature breakdown:")
print(f"  - EM:         {qc_report['feature_breakdown']['EM']}")
print(f"  - FLEN:       {qc_report['feature_breakdown']['FLEN']}")
print(f"  - NUCLEOSOME: {qc_report['feature_breakdown']['NUCLEOSOME']}")
print(f"\nTrain/Test split: {len(X_train_scaled)}/{len(X_test_scaled)} samples")
print(f"\n*** Data leakage prevention: All QC computed on TRAIN only ***")
