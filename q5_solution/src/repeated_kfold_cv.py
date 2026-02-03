#!/usr/bin/env python3
"""
Repeated K-Fold CV for Voting Ensemble.
Runs 5-fold CV with 5 different random seeds to get more stable estimates.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')
OUTPUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution')

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}

# Configuration
N_REPEATS = 5
N_SPLITS = 5
BASE_SEED = 42

# Load data
print("Loading data...")
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

# Load tuned params
phase2_tuned_path = RESULTS_DIR / 'phase2_tuning_results_v3.json'
xgb_rf_tuned_path = RESULTS_DIR / 'xgb_rf_tuning_v3.json'

default_xgb_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
default_rf_params = {'n_estimators': 200, 'max_depth': None}
default_lr_params = {}
default_svm_params = {'C': 1.0, 'gamma': 'scale'}

if phase2_tuned_path.exists():
    with open(phase2_tuned_path) as f:
        phase2_results = json.load(f)
    lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', default_lr_params)
    svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', default_svm_params)
else:
    lr_params = default_lr_params
    svm_params = default_svm_params

if xgb_rf_tuned_path.exists():
    with open(xgb_rf_tuned_path) as f:
        xgb_rf_results = json.load(f)
    xgb_params = xgb_rf_results.get('xgb', {}).get('best_params', default_xgb_params)
    rf_params = xgb_rf_results.get('rf', {}).get('best_params', default_rf_params)
else:
    xgb_params = default_xgb_params
    rf_params = default_rf_params

print(f"\nRunning Repeated {N_SPLITS}-Fold CV with {N_REPEATS} repeats = {N_SPLITS * N_REPEATS} total folds\n")

# Store results
all_fold_scores = []  # All fold scores across repeats
repeat_mean_scores = []  # Mean score per repeat
per_class_scores = {name: [] for name in LABEL_MAP.values()}
fold_details = []  # Store (repeat, fold, score)

for repeat in range(N_REPEATS):
    seed = BASE_SEED + repeat * 10
    print(f"\n{'='*50}")
    print(f"REPEAT {repeat + 1}/{N_REPEATS} (seed={seed})")
    print(f"{'='*50}")
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    repeat_scores = []
    all_preds = np.zeros(len(y_train), dtype=int)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        sample_weights = compute_sample_weight('balanced', y_tr)
        
        # Build models
        lr_kwargs = dict(lr_params)
        lr_kwargs.setdefault('class_weight', 'balanced')
        lr_model = LogisticRegression(**lr_kwargs, max_iter=2000, random_state=seed)
        
        svm_kwargs = dict(svm_params)
        svm_kwargs.setdefault('class_weight', 'balanced')
        svm_model = SVC(**svm_kwargs, random_state=seed, probability=True)
        
        rf_model = RandomForestClassifier(**rf_params, random_state=seed, class_weight='balanced', n_jobs=-1)
        xgb_model = XGBClassifier(**xgb_params, random_state=seed, eval_metric='mlogloss', n_jobs=-1)
        
        # Fit
        lr_model.fit(X_tr, y_tr)
        svm_model.fit(X_tr, y_tr)
        rf_model.fit(X_tr, y_tr)
        xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)
        
        # Predict probabilities
        proba_lr = lr_model.predict_proba(X_val)
        proba_svm = svm_model.predict_proba(X_val)
        proba_rf = rf_model.predict_proba(X_val)
        proba_xgb = xgb_model.predict_proba(X_val)
        
        # Soft voting
        avg_proba = (proba_lr + proba_svm + proba_rf + proba_xgb) / 4.0
        voting_pred = np.argmax(avg_proba, axis=1)
        
        f1 = f1_score(y_val, voting_pred, average='macro')
        repeat_scores.append(f1)
        all_fold_scores.append(f1)
        all_preds[val_idx] = voting_pred
        fold_details.append((repeat + 1, fold_idx + 1, f1))
        
        print(f"  Fold {fold_idx + 1}: F1 = {f1:.3f}")
    
    repeat_mean = np.mean(repeat_scores)
    repeat_mean_scores.append(repeat_mean)
    
    # Per-class F1 for this repeat
    f1_class = f1_score(y_train, all_preds, average=None)
    for i, name in LABEL_MAP.items():
        per_class_scores[name].append(f1_class[i])
    
    print(f"\n  Repeat {repeat + 1} Mean: {repeat_mean:.3f}")

# Final statistics
print("\n" + "="*60)
print("REPEATED K-FOLD CV RESULTS")
print("="*60)

overall_mean = np.mean(all_fold_scores)
overall_std = np.std(all_fold_scores)
overall_se = overall_std / np.sqrt(len(all_fold_scores))

print(f"\nOverall F1 Macro: {overall_mean:.4f} ± {overall_std:.4f}")
print(f"95% CI: [{overall_mean - 1.96*overall_se:.4f}, {overall_mean + 1.96*overall_se:.4f}]")
print(f"(Based on {N_SPLITS * N_REPEATS} folds)")

print("\nPer-repeat means:")
for i, score in enumerate(repeat_mean_scores):
    print(f"  Repeat {i+1}: {score:.4f}")

print("\nPer-class F1 (mean ± std across repeats):")
per_class_final = {}
for name in LABEL_MAP.values():
    mean_val = np.mean(per_class_scores[name])
    std_val = np.std(per_class_scores[name])
    per_class_final[name] = {'mean': mean_val, 'std': std_val}
    print(f"  {name}: {mean_val:.3f} ± {std_val:.3f}")

# Compare with single 5-fold
single_fold_path = RESULTS_DIR / 'voting_results_v3_notuned_xgb_rf.json'
if single_fold_path.exists():
    with open(single_fold_path) as f:
        single_results = json.load(f)
    single_mean = single_results['voting']['f1_macro_mean']
    single_std = single_results['voting']['f1_macro_std']
    single_fold_scores = single_results['voting']['fold_scores']
    print(f"\nComparison with Single 5-Fold CV:")
    print(f"  Single:   {single_mean:.4f} ± {single_std:.4f}")
    print(f"  Repeated: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"  Stability improvement: {(single_std - overall_std)/single_std * 100:.1f}%")

# ============================================
# CREATE VISUALIZATION
# ============================================
print("\nGenerating visualization...")

fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'Repeated {N_SPLITS}-Fold CV ({N_REPEATS} repeats = {N_SPLITS*N_REPEATS} folds)', 
             fontsize=14, fontweight='bold')

# Plot 1: All fold scores by repeat
ax1 = fig.add_subplot(2, 2, 1)
fold_data = np.array(all_fold_scores).reshape(N_REPEATS, N_SPLITS)
x_positions = np.arange(1, N_SPLITS + 1)
colors = plt.cm.Set2(np.linspace(0, 1, N_REPEATS))

for i in range(N_REPEATS):
    ax1.plot(x_positions, fold_data[i], 'o-', color=colors[i], 
             label=f'Repeat {i+1}', markersize=8, linewidth=2, alpha=0.8)

ax1.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.3f}')
ax1.fill_between(x_positions, overall_mean - overall_std, overall_mean + overall_std, 
                  alpha=0.2, color='red', label=f'±1 std: {overall_std:.3f}')
ax1.set_xlabel('Fold', fontsize=11)
ax1.set_ylabel('F1 Macro', fontsize=11)
ax1.set_title('F1 by Fold (All Repeats)', fontsize=12)
ax1.set_xticks(x_positions)
ax1.legend(loc='lower left', fontsize=8, ncol=2)
ax1.set_ylim(0.30, 0.65)
ax1.grid(alpha=0.3)

# Plot 2: Distribution of all fold scores
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(all_fold_scores, bins=12, color='#348ABD', edgecolor='black', alpha=0.7)
ax2.axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {overall_mean:.3f}')
ax2.axvline(x=overall_mean - overall_std, color='orange', linestyle=':', linewidth=2)
ax2.axvline(x=overall_mean + overall_std, color='orange', linestyle=':', linewidth=2,
            label=f'±1 std: {overall_std:.3f}')
ax2.set_xlabel('F1 Macro', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title(f'Distribution of F1 Scores (n={len(all_fold_scores)})', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Per-repeat mean comparison
ax3 = fig.add_subplot(2, 2, 3)
repeat_labels = [f'Repeat {i+1}' for i in range(N_REPEATS)]
bars = ax3.bar(repeat_labels, repeat_mean_scores, color=colors, edgecolor='black')
ax3.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
            label=f'Overall Mean: {overall_mean:.3f}')
ax3.set_ylabel('Mean F1 Macro', fontsize=11)
ax3.set_title('Mean F1 by Repeat', fontsize=12)
ax3.legend(fontsize=9)
ax3.set_ylim(0.40, 0.55)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, repeat_mean_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{score:.3f}', ha='center', va='bottom', fontsize=10)

# Plot 4: Per-class F1 with error bars
ax4 = fig.add_subplot(2, 2, 4)
class_names = list(LABEL_MAP.values())
means = [per_class_final[name]['mean'] for name in class_names]
stds = [per_class_final[name]['std'] for name in class_names]

x_pos = np.arange(len(class_names))
bars = ax4.bar(x_pos, means, yerr=stds, color='#4C78A8', edgecolor='black', 
               capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
ax4.set_xticks(x_pos)
ax4.set_xticklabels(class_names, fontsize=10)
ax4.set_ylabel('F1 Score', fontsize=11)
ax4.set_title('Per-Class F1 (mean ± std across repeats)', fontsize=12)
ax4.set_ylim(0, 0.8)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    ax4.text(i, mean + std + 0.02, f'{mean:.2f}', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'repeated_kfold_results.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved: {OUTPUT_DIR / 'repeated_kfold_results.png'}")

# Save results to JSON
results = {
    'n_repeats': N_REPEATS,
    'n_splits': N_SPLITS,
    'total_folds': N_SPLITS * N_REPEATS,
    'overall': {
        'f1_macro_mean': float(overall_mean),
        'f1_macro_std': float(overall_std),
        'f1_macro_se': float(overall_se),
        'ci_95': [float(overall_mean - 1.96*overall_se), float(overall_mean + 1.96*overall_se)]
    },
    'per_repeat_means': [float(s) for s in repeat_mean_scores],
    'all_fold_scores': [float(s) for s in all_fold_scores],
    'per_class': {name: {'mean': float(v['mean']), 'std': float(v['std'])} 
                  for name, v in per_class_final.items()},
    'fold_details': [{'repeat': r, 'fold': f, 'f1': float(s)} for r, f, s in fold_details]
}

with open(RESULTS_DIR / 'repeated_kfold_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'repeated_kfold_results.json'}")

print("\nDone!")
