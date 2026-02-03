#!/usr/bin/env python3
"""Plot Stage 2 (v3) feature selection summary as a 2x2 panel."""

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
METADATA_DIR = DATA_DIR / 'metadata'
OUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
with open(METADATA_DIR / 'selected_features_v3.json', 'r') as f:
    meta = json.load(f)

with open(METADATA_DIR / 'pca_models_v3.pkl', 'rb') as f:
    pca_models = pickle.load(f)

n_original = meta['n_features_original']
n_vip = meta['n_features_vip']
n_stable = meta['n_features_stable']
n_final = meta['n_features_final']

# Group counts for stable features
stable_features = meta['stable_selected']
counts = {
    'EM': sum(f.startswith('EM_') for f in stable_features),
    'FLEN': sum(f.startswith('FLEN_') for f in stable_features),
    'NUCLEOSOME': sum(f.startswith('NUC_') for f in stable_features),
}

# Group PCA components and variance
group_pcs = {g: info['n_components'] for g, info in pca_models.items()}
variance = {g: info['total_variance_explained'] for g, info in pca_models.items()}

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Stage 2 v3 Results Summary', fontsize=14, fontweight='bold')

# Panel 1: Feature reduction pipeline
ax = axes[0, 0]
steps = ['After QC', 'VIP', 'Stability', 'Group PCA']
vals = [n_original, n_vip, n_stable, n_final]
ax.plot(steps, vals, marker='o', linewidth=2, color='#1f77b4')
ax.fill_between(steps, vals, alpha=0.1, color='#1f77b4')
ax.set_title('Feature Reduction Pipeline')
ax.set_ylabel('Feature Count')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# Panel 2: Stable features by group
ax = axes[0, 1]
ax.bar(list(counts.keys()), list(counts.values()), color=['#4c78a8', '#f58518', '#54a24b'])
ax.set_title('Stable Features by Group (Top 30)')
ax.set_ylabel('Count')
ax.set_ylim(0, max(counts.values()) + 5)
for i, v in enumerate(counts.values()):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)

# Panel 3: Group PCA components
ax = axes[1, 0]
ax.bar(list(group_pcs.keys()), list(group_pcs.values()), color=['#4c78a8', '#f58518', '#54a24b'])
ax.set_title('Group PCA Components (Total = 15)')
ax.set_ylabel('PCs')
ax.set_ylim(0, max(group_pcs.values()) + 2)
for i, v in enumerate(group_pcs.values()):
    ax.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=9)

# Panel 4: Variance explained by group
ax = axes[1, 1]
labels = list(variance.keys())
vals = [variance[g] * 100 for g in labels]
ax.bar(labels, vals, color=['#4c78a8', '#f58518', '#54a24b'])
ax.set_title('Variance Explained by Group PCA')
ax.set_ylabel('Explained Variance (%)')
ax.set_ylim(0, max(vals) + 10)
for i, v in enumerate(vals):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

out_path = OUT_DIR / 'stage2_v3_results.png'
fig.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
