#!/usr/bin/env python3
"""Plot voting comparison: not tuned vs tuned XGB/RF (v3)."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')
OUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    'Not tuned XGB/RF': RESULTS_DIR / 'voting_results_v3_notuned_xgb_rf.json',
    'Tuned XGB/RF': RESULTS_DIR / 'voting_results_v3_tuned_xgb_rf.json'
}

loaded = {}
for label, path in paths.items():
    with open(path, 'r') as f:
        loaded[label] = json.load(f)

labels = list(loaded['Not tuned XGB/RF']['voting']['per_class'].keys())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Voting v3: Not Tuned vs Tuned XGB/RF', fontsize=14, fontweight='bold')

# Panel 1: Per-class F1 comparison
ax = axes[0, 0]
x = np.arange(len(labels))
width = 0.35
vals_a = [loaded['Not tuned XGB/RF']['voting']['per_class'][k] for k in labels]
vals_b = [loaded['Tuned XGB/RF']['voting']['per_class'][k] for k in labels]
ax.bar(x - width/2, vals_a, width, label='Not tuned', color='#e74c3c')
ax.bar(x + width/2, vals_b, width, label='Tuned', color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.8)
ax.set_title('Per-class F1')
ax.set_ylabel('F1')
ax.legend()

# Panel 2: Fold-by-fold F1
ax = axes[0, 1]
folds = np.arange(1, 6)
ax.plot(folds, loaded['Not tuned XGB/RF']['voting']['fold_scores'], 's-', color='#e74c3c', label='Not tuned', linewidth=2)
ax.plot(folds, loaded['Tuned XGB/RF']['voting']['fold_scores'], 'o-', color='#3498db', label='Tuned', linewidth=2)
ax.set_title('F1 by Fold')
ax.set_xlabel('Fold')
ax.set_ylabel('F1 Macro')
ax.set_xticks(folds)
ax.set_ylim(0.2, 0.7)
ax.legend()

# Panel 3: Confusion matrix (not tuned)
ax = axes[1, 0]
cm_a = np.array(loaded['Not tuned XGB/RF']['voting']['confusion_matrix'], dtype=float)
cm_a = cm_a / cm_a.sum(axis=1, keepdims=True)
sns.heatmap(cm_a, annot=True, fmt='.2f', cmap='Reds', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Confusion Matrix (Not tuned)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Panel 4: Confusion matrix (tuned)
ax = axes[1, 1]
cm_b = np.array(loaded['Tuned XGB/RF']['voting']['confusion_matrix'], dtype=float)
cm_b = cm_b / cm_b.sum(axis=1, keepdims=True)
sns.heatmap(cm_b, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Confusion Matrix (Tuned)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

out_path = OUT_DIR / 'voting_compare_v3.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
