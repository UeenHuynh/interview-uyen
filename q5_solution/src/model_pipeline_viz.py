#!/usr/bin/env python3
"""
cfDNA Classification Pipeline with Visualization

Generates 4 plots:
1. Per-class F1 scores
2. F1 by fold
3. Confusion matrix
4. Model vs Specialist comparison (if applicable)

Usage:
    python model_pipeline_viz.py --catboost
    python model_pipeline_viz.py --catboost-specialist --alpha 0.6
    python model_pipeline_viz.py --voting-catboost-specialist
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}
CLASS_NAMES = list(LABEL_MAP.values())
N_CLASSES = 6
CRC_IDX = 2
GASTRIC_IDX = 3

plt.style.use('seaborn-v0_8-whitegrid')


def load_data():
    X = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
    y = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values
    return X, y


# ============================================================
# Model Functions (return probabilities for specialist fusion)
# ============================================================
def get_model_proba(X_tr, y_tr, X_val, model_type='catboost'):
    """Train model and return probabilities."""
    sample_weights = compute_sample_weight('balanced', y_tr)
    
    if model_type == 'lr':
        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)
    
    elif model_type == 'svm':
        model = SVC(C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)
    
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)
    
    elif model_type == 'xgb':
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        return model.predict_proba(X_val)
    
    elif model_type == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
                                  loss_function='MultiClass', auto_class_weights='Balanced',
                                  random_state=RANDOM_STATE, verbose=False)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)
    
    elif model_type == 'voting':
        lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
        svm = SVC(C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
        lr.fit(X_tr, y_tr)
        svm.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr, sample_weight=sample_weights)
        return (lr.predict_proba(X_val) + svm.predict_proba(X_val) + 
                rf.predict_proba(X_val) + xgb.predict_proba(X_val)) / 4
    
    elif model_type == 'voting_catboost':
        from catboost import CatBoostClassifier
        lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
        svm = SVC(C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        cb = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
                               loss_function='MultiClass', auto_class_weights='Balanced',
                               random_state=RANDOM_STATE, verbose=False)
        lr.fit(X_tr, y_tr)
        svm.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        cb.fit(X_tr, y_tr)
        return (lr.predict_proba(X_val) + svm.predict_proba(X_val) + 
                rf.predict_proba(X_val) + cb.predict_proba(X_val)) / 4


def get_specialist_proba(X_tr, y_tr, X_val, target_class):
    """Train binary specialist and return P(target_class)."""
    y_binary = (y_tr == target_class).astype(int)
    weights = compute_sample_weight('balanced', y_binary)
    model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1,
                         random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
    model.fit(X_tr, y_binary, sample_weight=weights)
    return model.predict_proba(X_val)[:, 1]


def fuse_proba(general_proba, crc_proba, gastric_proba, alpha):
    """Fuse base model proba with specialist proba."""
    fused = general_proba.copy()
    fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba
    fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused


# ============================================================
# Main Evaluation Function
# ============================================================
def evaluate_model(X, y, model_type, use_specialist=False, alpha=0.8):
    """
    Run 5-fold CV and collect all metrics for visualization.
    
    Returns dict with:
    - fold_scores, fold_scores_base (if specialist)
    - all_preds, all_preds_base (if specialist)
    - per_class_f1
    - confusion_matrix
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    fold_scores = []
    fold_scores_base = []
    fold_accuracy = []
    fold_auc = []
    all_preds = np.zeros(len(y), dtype=int)
    all_preds_base = np.zeros(len(y), dtype=int)
    all_proba = np.zeros((len(y), N_CLASSES))
    
    name = model_type.replace('_', ' ').title()
    if use_specialist:
        name += f' + Specialist (α={alpha})'
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Base model prediction
        base_proba = get_model_proba(X_tr, y_tr, X_val, model_type)
        base_pred = np.argmax(base_proba, axis=1)
        base_f1 = f1_score(y_val, base_pred, average='macro')
        fold_scores_base.append(base_f1)
        all_preds_base[val_idx] = base_pred
        
        if use_specialist:
            # Get specialist probabilities
            crc_proba = get_specialist_proba(X_tr, y_tr, X_val, CRC_IDX)
            gastric_proba = get_specialist_proba(X_tr, y_tr, X_val, GASTRIC_IDX)
            
            # Fuse probabilities
            fused_proba = fuse_proba(base_proba, crc_proba, gastric_proba, alpha)
            final_pred = np.argmax(fused_proba, axis=1)
            final_f1 = f1_score(y_val, final_pred, average='macro')
            
            fold_scores.append(final_f1)
            all_preds[val_idx] = final_pred
            all_proba[val_idx] = fused_proba
            
            # Accuracy and AUC
            acc = accuracy_score(y_val, final_pred)
            y_val_bin = label_binarize(y_val, classes=list(range(N_CLASSES)))
            auc = roc_auc_score(y_val_bin, fused_proba, average='macro', multi_class='ovr')
            fold_accuracy.append(acc)
            fold_auc.append(auc)
            
            print(f"  Fold {fold_idx+1}: F1={final_f1:.3f} | Acc={acc:.3f} | AUC={auc:.3f} (base F1={base_f1:.3f})")
        else:
            fold_scores.append(base_f1)
            all_preds[val_idx] = base_pred
            all_proba[val_idx] = base_proba
            
            # Accuracy and AUC
            acc = accuracy_score(y_val, base_pred)
            y_val_bin = label_binarize(y_val, classes=list(range(N_CLASSES)))
            auc = roc_auc_score(y_val_bin, base_proba, average='macro', multi_class='ovr')
            fold_accuracy.append(acc)
            fold_auc.append(auc)
            
            print(f"  Fold {fold_idx+1}: F1={base_f1:.3f} | Acc={acc:.3f} | AUC={auc:.3f}")
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    mean_acc = np.mean(fold_accuracy)
    mean_auc = np.mean(fold_auc)
    per_class = f1_score(y, all_preds, average=None)
    cm = confusion_matrix(y, all_preds)
    overall_acc = accuracy_score(y, all_preds)
    
    print(f"\n  → F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  → Accuracy: {mean_acc:.3f}")
    print(f"  → AUC-ROC:  {mean_auc:.3f}")
    
    result = {
        'name': name,
        'model_type': model_type,
        'use_specialist': use_specialist,
        'alpha': alpha if use_specialist else None,
        # Primary metric
        'f1_macro_mean': float(mean_f1),
        'f1_macro_std': float(std_f1),
        # Secondary metrics
        'accuracy_mean': float(mean_acc),
        'accuracy_overall': float(overall_acc),
        'auc_roc_mean': float(mean_auc),
        # Per-fold
        'fold_f1': [float(s) for s in fold_scores],
        'fold_accuracy': [float(a) for a in fold_accuracy],
        'fold_auc': [float(a) for a in fold_auc],
        # Per-class
        'per_class_f1': {LABEL_MAP[i]: float(per_class[i]) for i in range(N_CLASSES)},
        'confusion_matrix': cm.tolist(),
        'all_preds': all_preds,
        'all_proba': all_proba,
        'y_true': y
    }
    
    if use_specialist:
        result['fold_f1_base'] = [float(s) for s in fold_scores_base]
        result['all_preds_base'] = all_preds_base
        result['per_class_f1_base'] = {LABEL_MAP[i]: float(v) for i, v in 
                                       enumerate(f1_score(y, all_preds_base, average=None))}
        result['confusion_matrix_base'] = confusion_matrix(y, all_preds_base).tolist()
    
    return result


# ============================================================
# Visualization Functions
# ============================================================
def plot_results(result, output_prefix):
    """Generate 4 plots and save them."""
    
    name = result['name']
    use_specialist = result['use_specialist']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{name}\nF1: {result["f1_macro_mean"]:.3f} | Acc: {result["accuracy_mean"]:.3f} | AUC: {result["auc_roc_mean"]:.3f}', 
                 fontsize=14, fontweight='bold')
    
    # ========== 1. Per-class F1 ==========
    ax1 = axes[0, 0]
    x = np.arange(N_CLASSES)
    per_class = [result['per_class_f1'][c] for c in CLASS_NAMES]
    
    if use_specialist:
        per_class_base = [result['per_class_f1_base'][c] for c in CLASS_NAMES]
        width = 0.35
        bars1 = ax1.bar(x - width/2, per_class_base, width, label='Base Model', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, per_class, width, label='+ Specialist', color='#2ecc71', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars1, per_class_base):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, per_class):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=8)
        
        # Highlight specialist classes
        for i in [CRC_IDX, GASTRIC_IDX]:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='yellow')
        ax1.legend()
    else:
        bars = ax1.bar(x, per_class, color='#3498db', alpha=0.8)
        for bar, val in zip(bars, per_class):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('F1 Score')
    ax1.set_xlabel('Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.set_title('Per-class F1 Scores')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=result['f1_macro_mean'], color='red', linestyle='--', alpha=0.5, label='Mean')
    
    # ========== 2. F1 by Fold ==========
    ax2 = axes[0, 1]
    folds = np.arange(1, 6)
    
    if use_specialist:
        ax2.plot(folds, result['fold_f1_base'], 'o-', label='Base Model', 
                color='#3498db', linewidth=2, markersize=8)
        ax2.plot(folds, result['fold_f1'], 's-', label='+ Specialist', 
                color='#2ecc71', linewidth=2, markersize=8)
        ax2.legend()
    else:
        ax2.plot(folds, result['fold_f1'], 'o-', color='#3498db', linewidth=2, markersize=10)
    
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('F1 Macro')
    ax2.set_xticks(folds)
    ax2.set_title('F1 Score by Fold')
    ax2.set_ylim(0.2, 0.7)
    ax2.axhline(y=result['f1_macro_mean'], color='red', linestyle='--', alpha=0.5)
    
    # ========== 3. Confusion Matrix (Base or Final) ==========
    ax3 = axes[1, 0]
    cm = np.array(result['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    title = 'Confusion Matrix'
    if use_specialist:
        title += ' (with Specialist)'
    ax3.set_title(title)
    
    # ========== 4. Model vs Specialist Comparison / Base CM ==========
    ax4 = axes[1, 1]
    
    if use_specialist:
        # Show improvement per class
        improvements = []
        for c in CLASS_NAMES:
            imp = result['per_class_f1'][c] - result['per_class_f1_base'][c]
            improvements.append(imp)
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax4.barh(CLASS_NAMES, improvements, color=colors, alpha=0.8)
        
        ax4.axvline(x=0, color='black', linewidth=0.5)
        ax4.set_xlabel('F1 Improvement')
        ax4.set_title('Specialist Improvement by Class')
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            x_pos = bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005
            ha = 'left' if val >= 0 else 'right'
            ax4.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
                    ha=ha, va='center', fontsize=9)
        
        # Highlight specialist classes
        for i, c in enumerate(CLASS_NAMES):
            if c in ['CRC', 'Gastric']:
                ax4.get_yticklabels()[i].set_fontweight('bold')
    else:
        # Show base confusion matrix (normalized)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', ax=ax4,
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        ax4.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = FIGURES_DIR / f'{output_prefix}_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved figure: {fig_path}")
    return fig_path


def save_results(result, output_prefix):
    """Save results to JSON."""
    # Remove numpy arrays for JSON serialization
    result_json = {k: v for k, v in result.items() 
                  if k not in ['all_preds', 'all_preds_base', 'y_true', 'all_proba']}
    result_json['timestamp'] = datetime.now().isoformat()
    
    json_path = RESULTS_DIR / f'{output_prefix}_results.json'
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print(f"✓ Saved results: {json_path}")
    return json_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='cfDNA Pipeline with Visualization')
    
    # Model selection
    parser.add_argument('--lr', action='store_true', help='Logistic Regression')
    parser.add_argument('--svm', action='store_true', help='SVM')
    parser.add_argument('--rf', action='store_true', help='Random Forest')
    parser.add_argument('--xgb', action='store_true', help='XGBoost')
    parser.add_argument('--catboost', action='store_true', help='CatBoost')
    parser.add_argument('--voting', action='store_true', help='Voting (LR+SVM+RF+XGB)')
    parser.add_argument('--voting-catboost', action='store_true', help='Voting (LR+SVM+RF+CatBoost)')
    
    # Specialist options
    parser.add_argument('--specialist', action='store_true', help='Add CRC/Gastric specialists')
    parser.add_argument('--catboost-specialist', action='store_true', help='CatBoost + Specialists')
    parser.add_argument('--voting-specialist', action='store_true', help='Voting + Specialists')
    parser.add_argument('--voting-catboost-specialist', action='store_true', help='Voting(CatBoost) + Specialists')
    parser.add_argument('--alpha', type=float, default=0.8, help='Fusion weight α')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧬 cfDNA Classification Pipeline with Visualization")
    print("="*60)
    
    X, y = load_data()
    print(f"Data: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Determine model type and specialist flag
    model_type = None
    use_specialist = False
    
    if args.lr:
        model_type = 'lr'
    elif args.svm:
        model_type = 'svm'
    elif args.rf:
        model_type = 'rf'
    elif args.xgb:
        model_type = 'xgb'
    elif args.catboost or args.catboost_specialist:
        model_type = 'catboost'
        use_specialist = args.catboost_specialist or args.specialist
    elif args.voting or args.voting_specialist:
        model_type = 'voting'
        use_specialist = args.voting_specialist or args.specialist
    elif args.voting_catboost or args.voting_catboost_specialist:
        model_type = 'voting_catboost'
        use_specialist = args.voting_catboost_specialist or args.specialist
    else:
        # Default: catboost
        model_type = 'catboost'
    
    if args.specialist:
        use_specialist = True
    
    # Run evaluation
    result = evaluate_model(X, y, model_type, use_specialist, args.alpha)
    
    # Generate output prefix
    output_prefix = model_type
    if use_specialist:
        output_prefix += f'_specialist_alpha{args.alpha}'
    
    # Save results and plot
    save_results(result, output_prefix)
    plot_results(result, output_prefix)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
