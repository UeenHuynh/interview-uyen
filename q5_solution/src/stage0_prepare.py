#!/usr/bin/env python3
"""
Stage 0: Parquet Standardization & Metadata
- Transpose datasets (features as columns, samples as rows)
- Extract labels from sample names
- Prefix columns with group names (EM_, FLEN_, NUC_)
- Save to Parquet format with metadata
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from collections import Counter

# ============================================================
# Configuration
# ============================================================
RAW_DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/interview_data/Supplement_datasets')
OUTPUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')

# ============================================================
# 1. Load & Transpose Data
# ============================================================
print("=" * 60)
print("Stage 0: Loading and preparing data...")
print("=" * 60)

# Load CSV files (original format: features as rows, samples as columns)
em_raw = pd.read_csv(RAW_DATA_DIR / 'EM.csv', index_col=0)
flen_raw = pd.read_csv(RAW_DATA_DIR / 'FLEN.csv', index_col=0)
nuc_raw = pd.read_csv(RAW_DATA_DIR / 'NUCLEOSOME.csv', index_col=0)

# Transpose: samples as rows, features as columns
em = em_raw.T
flen = flen_raw.T
nuc = nuc_raw.T

print(f"EM shape (after transpose):          {em.shape}")
print(f"FLEN shape (after transpose):        {flen.shape}")
print(f"NUCLEOSOME shape (after transpose):  {nuc.shape}")

# ============================================================
# 2. Extract Labels
# ============================================================
label_map = {
    "Control": 0,
    "Breast": 1,
    "CRC": 2,
    "Gastric": 3,
    "Liver": 4,
    "Lung": 5
}

# Extract class from sample names (e.g., "Control_1" -> "Control")
sample_names = list(em.index)
labels = [s.split("_")[0] for s in sample_names]
y = pd.Series([label_map[label] for label in labels], index=em.index, name="label")

print(f"\nLabel distribution:")
label_counts = Counter(labels)
for cls, count in sorted(label_counts.items()):
    print(f"  {cls}: {count} samples ({count/len(labels)*100:.1f}%)")

# ============================================================
# 3. Prefix Columns & Combine
# ============================================================
# Add prefixes to distinguish feature groups
em.columns = ["EM_" + str(c) for c in em.columns]
flen.columns = ["FLEN_" + str(c) for c in flen.columns]
nuc.columns = ["NUC_" + str(c) for c in nuc.columns]

# Combine all features
X = pd.concat([em, flen, nuc], axis=1)
print(f"\nCombined shape: {X.shape}")
print(f"  - EM features:         {len([c for c in X.columns if c.startswith('EM_')])}")
print(f"  - FLEN features:       {len([c for c in X.columns if c.startswith('FLEN_')])}")
print(f"  - NUCLEOSOME features: {len([c for c in X.columns if c.startswith('NUC_')])}")

# ============================================================
# 4. Save Parquet Files
# ============================================================
processed_dir = OUTPUT_DIR / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

# Save combined features
X.to_parquet(processed_dir / 'cleaned_data.parquet', compression='snappy')
print(f"\nSaved: {processed_dir / 'cleaned_data.parquet'}")

# Save labels
y.to_frame().to_parquet(processed_dir / 'labels.parquet', compression='snappy')
print(f"Saved: {processed_dir / 'labels.parquet'}")

# ============================================================
# 5. Save Metadata
# ============================================================
metadata_dir = OUTPUT_DIR / 'metadata'
metadata_dir.mkdir(parents=True, exist_ok=True)

# Feature groups mapping
feature_groups = {
    "FLEN": [c for c in X.columns if c.startswith("FLEN_")],
    "EM": [c for c in X.columns if c.startswith("EM_")],
    "NUCLEOSOME": [c for c in X.columns if c.startswith("NUC_")]
}

# Complete metadata
metadata = {
    "n_samples": int(len(X)),
    "n_features_original": int(X.shape[1]),
    "feature_groups": feature_groups,
    "label_map": label_map,
    "reverse_label_map": {v: k for k, v in label_map.items()},
    "class_counts": {k: int(v) for k, v in label_counts.items()},
    "compression": "snappy",
    "created": pd.Timestamp.now().isoformat()
}

# Save as JSON
with open(metadata_dir / 'feature_groups.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved: {metadata_dir / 'feature_groups.json'}")

# Also save preprocessing config
preprocessing_config = {
    "raw_files": {
        "EM": str(RAW_DATA_DIR / 'EM.csv'),
        "FLEN": str(RAW_DATA_DIR / 'FLEN.csv'),
        "NUCLEOSOME": str(RAW_DATA_DIR / 'NUCLEOSOME.csv')
    },
    "label_encoding": label_map,
    "feature_prefixes": {
        "EM": "EM_",
        "FLEN": "FLEN_",
        "NUCLEOSOME": "NUC_"
    },
    "output_format": "parquet",
    "compression": "snappy"
}

with open(metadata_dir / 'preprocessing_config.json', 'w') as f:
    json.dump(preprocessing_config, f, indent=2)
print(f"Saved: {metadata_dir / 'preprocessing_config.json'}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Stage 0 Complete!")
print("=" * 60)
print(f"Total samples:  {X.shape[0]}")
print(f"Total features: {X.shape[1]}")
print(f"Output files:")
print(f"  - data/processed/cleaned_data.parquet")
print(f"  - data/processed/labels.parquet")
print(f"  - data/metadata/feature_groups.json")
print(f"  - data/metadata/preprocessing_config.json")
