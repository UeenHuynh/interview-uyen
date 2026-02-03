#!/usr/bin/env python3
"""Plot voting-only results for v3 features."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')
OUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / 'voting_results_v3.json', 'r') as f:
    data = json.load(f)

per_class = data['voting']['per_class']
fold_scores = data['voting']['fold_scores']
cm = np.array(data['voting']['confusion_matrix'], dtype=float)

labels = list(per_class.keys())
per_class_vals = [per_class[k] for k in labels]

# Normalize confusion matrix by row
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Per-class F1 bar
ax = axes[0]
ax.bar(labels, per_class_vals, color='#e74c3c')
ax.set_ylim(0, 0.8)
ax.set_title('Voting Per-class F1')
ax.set_ylabel('F1')
for i, v in enumerate(per_class_vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

# Confusion matrix heatmap
ax = axes[1]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Voting Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Fold-by-fold line
ax = axes[2]
folds = np.arange(1, len(fold_scores) + 1)
ax.plot(folds, fold_scores, 's-', color='#e74c3c', linewidth=2)
ax.set_title('Voting F1 by Fold')
ax.set_xlabel('Fold')
ax.set_ylabel('F1 Macro')
ax.set_xticks(folds)
ax.set_ylim(0.2, 0.7)

plt.tight_layout()

out_path = OUT_DIR / 'voting_results_v3.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
