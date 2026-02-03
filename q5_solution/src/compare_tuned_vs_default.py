#!/usr/bin/env python3
"""
Compare Option 6 performance with:
1. Default XGB/RF params (hardcoded)
2. Tuned XGB/RF params (from xgb_rf_tuning_v3.json)

Generate comparison charts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/data')
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/results')
OUTPUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}
CRC_IDX = 2
GASTRIC_IDX = 3

ALPHA_VALUES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

print("=" * 70)
print("Comparison: Default XGB/RF vs Tuned XGB/RF")
print("=" * 70)

# ============================================================
# Load Data
# ============================================================
X_train = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values

print(f"\nData: {X_train.shape[0]} samples × {X_train.shape[1]} features")

# ============================================================
# Define Parameter Sets
# ============================================================

# Load phase2 tuned params for LR/SVM (shared)
phase2_tuned_path = RESULTS_DIR / 'phase2_tuning_results_v3.json'
with open(phase2_tuned_path) as f:
    phase2_results = json.load(f)
lr_params = phase2_results.get('LogisticRegression', {}).get('best_params', {})
svm_params = phase2_results.get('SVM_RBF', {}).get('best_params', {'C': 1.0, 'gamma': 'scale'})

# Config 1: DEFAULT XGB/RF (hardcoded)
DEFAULT_XGB = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
DEFAULT_RF = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}

# Config 2: TUNED XGB/RF (from json)
xgb_rf_tuned_path = RESULTS_DIR / 'xgb_rf_tuning_v3.json'
with open(xgb_rf_tuned_path) as f:
    xgb_rf_results = json.load(f)
TUNED_XGB = xgb_rf_results.get('xgb', {}).get('best_params', DEFAULT_XGB)
TUNED_RF = xgb_rf_results.get('rf', {}).get('best_params', DEFAULT_RF)

print("\n--- Parameter Comparison ---")
print(f"DEFAULT XGB: {DEFAULT_XGB}")
print(f"TUNED XGB:   {TUNED_XGB}")
print(f"\nDEFAULT RF:  {DEFAULT_RF}")
print(f"TUNED RF:    {TUNED_RF}")

# ============================================================
# Helper Functions
# ============================================================

def create_voting_models(xgb_params, rf_params):
    lr_kwargs = dict(lr_params)
    lr_kwargs.setdefault('class_weight', 'balanced')
    svm_kwargs = dict(svm_params)
    svm_kwargs.setdefault('class_weight', 'balanced')
    
    return {
        'lr': LogisticRegression(**lr_kwargs, max_iter=2000, random_state=RANDOM_STATE),
        'svm': SVC(**svm_kwargs, random_state=RANDOM_STATE, probability=True),
        'rf': RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        'xgb': XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
    }

def create_specialist():
    return XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, subsample=0.8, 
                        random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)

def train_voting(models, X, y, sample_weights=None):
    for name, model in models.items():
        if name == 'xgb' and sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
    return models

def predict_voting_proba(models, X):
    probas = [model.predict_proba(X) for model in models.values()]
    return np.mean(probas, axis=0)

def fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=0.5):
    fused = general_proba.copy()
    fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba
    fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused

def run_alpha_experiment(config_name, xgb_params, rf_params):
    """Run full alpha tuning experiment with given XGB/RF params."""
    print(f"\n{'='*50}")
    print(f"Running: {config_name}")
    print(f"{'='*50}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    for alpha in ALPHA_VALUES:
        specialist_scores = []
        crc_scores = []
        gastric_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            sample_weights = compute_sample_weight('balanced', y_tr)
            
            # Train Voting
            voting_models = create_voting_models(xgb_params, rf_params)
            voting_models = train_voting(voting_models, X_tr, y_tr, sample_weights)
            general_proba = predict_voting_proba(voting_models, X_val)
            
            # Train CRC Specialist
            y_crc_binary = (y_tr == CRC_IDX).astype(int)
            crc_weights = compute_sample_weight('balanced', y_crc_binary)
            crc_specialist = create_specialist()
            crc_specialist.fit(X_tr, y_crc_binary, sample_weight=crc_weights)
            crc_proba = crc_specialist.predict_proba(X_val)[:, 1]
            
            # Train Gastric Specialist
            y_gastric_binary = (y_tr == GASTRIC_IDX).astype(int)
            gastric_weights = compute_sample_weight('balanced', y_gastric_binary)
            gastric_specialist = create_specialist()
            gastric_specialist.fit(X_tr, y_gastric_binary, sample_weight=gastric_weights)
            gastric_proba = gastric_specialist.predict_proba(X_val)[:, 1]
            
            # Fuse
            fused_proba = fuse_probabilities(general_proba, crc_proba, gastric_proba, alpha=alpha)
            specialist_pred = np.argmax(fused_proba, axis=1)
            
            # Evaluate
            specialist_f1 = f1_score(y_val, specialist_pred, average='macro')
            specialist_scores.append(specialist_f1)
            
            specialist_f1_class = f1_score(y_val, specialist_pred, average=None)
            crc_scores.append(specialist_f1_class[CRC_IDX])
            gastric_scores.append(specialist_f1_class[GASTRIC_IDX])
        
        results[alpha] = {
            'f1_macro_mean': np.mean(specialist_scores),
            'f1_macro_std': np.std(specialist_scores),
            'crc_mean': np.mean(crc_scores),
            'gastric_mean': np.mean(gastric_scores),
            'fold_scores': specialist_scores
        }
        
        print(f"  α={alpha}: F1={results[alpha]['f1_macro_mean']:.4f} ± {results[alpha]['f1_macro_std']:.4f}")
    
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda a: results[a]['f1_macro_mean'])
    best_f1 = results[best_alpha]['f1_macro_mean']
    
    print(f"\n  🏆 Best: α={best_alpha}, F1={best_f1:.4f}")
    
    return results, best_alpha, best_f1

# ============================================================
# Run Experiments
# ============================================================

# Experiment 1: Default params
default_results, default_best_alpha, default_best_f1 = run_alpha_experiment(
    "DEFAULT XGB/RF", DEFAULT_XGB, DEFAULT_RF
)

# Experiment 2: Tuned params
tuned_results, tuned_best_alpha, tuned_best_f1 = run_alpha_experiment(
    "TUNED XGB/RF", TUNED_XGB, TUNED_RF
)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"\n{'Config':<20} {'Best α':<10} {'Best F1':<15}")
print("-" * 45)
print(f"{'DEFAULT XGB/RF':<20} {default_best_alpha:<10} {default_best_f1:.4f}")
print(f"{'TUNED XGB/RF':<20} {tuned_best_alpha:<10} {tuned_best_f1:.4f}")

diff = tuned_best_f1 - default_best_f1
print(f"\nDifference: {diff:+.4f} ({diff/default_best_f1*100:+.2f}%)")

if tuned_best_f1 > default_best_f1:
    print("✅ TUNED params are BETTER")
else:
    print("⚠️ DEFAULT params are better or equal")

# ============================================================
# Create Visualization
# ============================================================
print("\nGenerating comparison chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparison: Default XGB/RF vs Tuned XGB/RF', fontsize=14, fontweight='bold')

# Plot 1: F1 by Alpha
ax1 = axes[0, 0]
alphas = ALPHA_VALUES
default_f1s = [default_results[a]['f1_macro_mean'] for a in alphas]
tuned_f1s = [tuned_results[a]['f1_macro_mean'] for a in alphas]
default_stds = [default_results[a]['f1_macro_std'] for a in alphas]
tuned_stds = [tuned_results[a]['f1_macro_std'] for a in alphas]

ax1.errorbar(alphas, default_f1s, yerr=default_stds, marker='o', capsize=5, 
             linewidth=2, markersize=8, label='Default XGB/RF', color='#E24A33')
ax1.errorbar(alphas, tuned_f1s, yerr=tuned_stds, marker='s', capsize=5, 
             linewidth=2, markersize=8, label='Tuned XGB/RF', color='#348ABD')
ax1.axvline(x=default_best_alpha, color='#E24A33', linestyle='--', alpha=0.5)
ax1.axvline(x=tuned_best_alpha, color='#348ABD', linestyle='--', alpha=0.5)
ax1.set_xlabel('Alpha (α)')
ax1.set_ylabel('F1 Macro')
ax1.set_title('F1 Macro by Alpha')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.42, 0.52)

# Plot 2: CRC F1 by Alpha
ax2 = axes[0, 1]
default_crc = [default_results[a]['crc_mean'] for a in alphas]
tuned_crc = [tuned_results[a]['crc_mean'] for a in alphas]

ax2.plot(alphas, default_crc, 'o-', linewidth=2, markersize=8, label='Default', color='#E24A33')
ax2.plot(alphas, tuned_crc, 's-', linewidth=2, markersize=8, label='Tuned', color='#348ABD')
ax2.set_xlabel('Alpha (α)')
ax2.set_ylabel('CRC F1')
ax2.set_title('CRC F1 by Alpha')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Gastric F1 by Alpha
ax3 = axes[1, 0]
default_gastric = [default_results[a]['gastric_mean'] for a in alphas]
tuned_gastric = [tuned_results[a]['gastric_mean'] for a in alphas]

ax3.plot(alphas, default_gastric, 'o-', linewidth=2, markersize=8, label='Default', color='#E24A33')
ax3.plot(alphas, tuned_gastric, 's-', linewidth=2, markersize=8, label='Tuned', color='#348ABD')
ax3.set_xlabel('Alpha (α)')
ax3.set_ylabel('Gastric F1')
ax3.set_title('Gastric F1 by Alpha')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Best Alpha Comparison Bar Chart
ax4 = axes[1, 1]
x = np.arange(2)
width = 0.35

# Get results at best alpha for each config
default_at_best = default_results[default_best_alpha]
tuned_at_best = tuned_results[tuned_best_alpha]

metrics = ['F1 Macro', 'CRC F1', 'Gastric F1']
default_vals = [default_at_best['f1_macro_mean'], default_at_best['crc_mean'], default_at_best['gastric_mean']]
tuned_vals = [tuned_at_best['f1_macro_mean'], tuned_at_best['crc_mean'], tuned_at_best['gastric_mean']]

x_pos = np.arange(len(metrics))
bars1 = ax4.bar(x_pos - width/2, default_vals, width, label=f'Default (α={default_best_alpha})', color='#E24A33')
bars2 = ax4.bar(x_pos + width/2, tuned_vals, width, label=f'Tuned (α={tuned_best_alpha})', color='#348ABD')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.set_ylabel('Score')
ax4.set_title('Best Results Comparison')
ax4.legend()
ax4.set_ylim(0, 0.6)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, default_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
             ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, tuned_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'tuned_vs_default_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved: {OUTPUT_DIR / 'tuned_vs_default_comparison.png'}")

# Save results
comparison_results = {
    'default': {
        'xgb_params': DEFAULT_XGB,
        'rf_params': DEFAULT_RF,
        'best_alpha': default_best_alpha,
        'best_f1': default_best_f1,
        'results': {str(k): v for k, v in default_results.items()}
    },
    'tuned': {
        'xgb_params': TUNED_XGB,
        'rf_params': TUNED_RF,
        'best_alpha': tuned_best_alpha,
        'best_f1': tuned_best_f1,
        'results': {str(k): v for k, v in tuned_results.items()}
    },
    'comparison': {
        'f1_difference': diff,
        'percent_change': diff/default_best_f1*100,
        'winner': 'tuned' if tuned_best_f1 > default_best_f1 else 'default'
    }
}

with open(RESULTS_DIR / 'tuned_vs_default_comparison.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'tuned_vs_default_comparison.json'}")

print("\nDone!")
