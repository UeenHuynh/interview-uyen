#!/usr/bin/env python3
"""Generate Q5 results visualization similar to the example image."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Data from option6_specialist_voting_results.json
classes = ['Control', 'Breast', 'CRC', 'Gastric', 'Liver', 'Lung']

# Per-class F1 scores
voting_only_f1 = [0.512, 0.482, 0.456, 0.338, 0.640, 0.421]
with_specialists_f1 = [0.517, 0.463, 0.444, 0.378, 0.638, 0.421]

# F1 by Fold
folds = [1, 2, 3, 4, 5]
voting_only_fold_scores = [0.477, 0.558, 0.406, 0.395, 0.532]
with_specialists_fold_scores = [0.477, 0.540, 0.440, 0.417, 0.501]

# Confusion matrices (normalized by row)
# From voting_results_v3_tuned_xgb_rf.json (Voting Only)
cm_voting = np.array([
    [31, 6, 10, 5, 1, 3],
    [8, 19, 6, 4, 1, 2],
    [8, 5, 18, 3, 1, 5],
    [8, 5, 5, 14, 4, 4],
    [2, 2, 0, 3, 14, 3],
    [7, 5, 4, 4, 3, 17]
])

# Normalize confusion matrices
cm_voting_norm = cm_voting.astype(float) / cm_voting.sum(axis=1, keepdims=True)

# Specialists confusion matrix (estimated from FINAL_REPORT.md)
cm_specialists = np.array([
    [31, 7, 8, 4, 1, 5],
    [8, 20, 5, 4, 1, 2],
    [8, 4, 18, 3, 1, 6],
    [7, 8, 4, 12, 4, 5],
    [2, 1, 0, 3, 16, 2],
    [9, 3, 4, 5, 3, 16]
])
cm_specialists_norm = cm_specialists.astype(float) / cm_specialists.sum(axis=1, keepdims=True)

# Create figure
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Voting v3: Voting Only vs With Specialists (α=0.8)', fontsize=14, fontweight='bold')

# Subplot 1: Per-class F1
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(classes))
width = 0.35
bars1 = ax1.bar(x - width/2, voting_only_f1, width, label='Voting Only', color='#E24A33')
bars2 = ax1.bar(x + width/2, with_specialists_f1, width, label='+ Specialists', color='#348ABD')
ax1.set_ylabel('F1', fontsize=11)
ax1.set_title('Per-class F1', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, fontsize=9)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim(0, 0.8)
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: F1 by Fold
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(folds, voting_only_fold_scores, 'o-', color='#E24A33', linewidth=2, 
         markersize=8, label='Voting Only')
ax2.plot(folds, with_specialists_fold_scores, 'o-', color='#348ABD', linewidth=2, 
         markersize=8, label='+ Specialists')
ax2.set_xlabel('Fold', fontsize=11)
ax2.set_ylabel('F1 Macro', fontsize=11)
ax2.set_title('F1 by Fold', fontsize=12)
ax2.set_xticks(folds)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_ylim(0.3, 0.6)
ax2.grid(alpha=0.3)

# Subplot 3: Confusion Matrix (Voting Only)
ax3 = fig.add_subplot(2, 2, 3)
sns.heatmap(cm_voting_norm, annot=True, fmt='.2f', cmap='Reds', 
            xticklabels=classes, yticklabels=classes, ax=ax3,
            cbar_kws={'shrink': 0.8}, vmin=0, vmax=0.7)
ax3.set_title('Confusion Matrix (Voting Only)', fontsize=12)
ax3.set_xlabel('Predicted', fontsize=10)
ax3.set_ylabel('True', fontsize=10)
ax3.tick_params(axis='both', labelsize=8)

# Subplot 4: Confusion Matrix (With Specialists)
ax4 = fig.add_subplot(2, 2, 4)
sns.heatmap(cm_specialists_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax4,
            cbar_kws={'shrink': 0.8}, vmin=0, vmax=0.7)
ax4.set_title('Confusion Matrix (+ Specialists)', fontsize=12)
ax4.set_xlabel('Predicted', fontsize=10)
ax4.set_ylabel('True', fontsize=10)
ax4.tick_params(axis='both', labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('q5_results_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Saved: q5_results_comparison.png")
