#!/usr/bin/env python3
"""
Unified cfDNA Classification Pipeline

Run individual models, voting, or specialist ensembles.

Usage:
    # Individual models
    python model_pipeline.py --lr
    python model_pipeline.py --svm
    python model_pipeline.py --rf
    python model_pipeline.py --xgb
    python model_pipeline.py --catboost
    
    # Voting ensembles
    python model_pipeline.py --voting              # LR+SVM+RF+XGB
    python model_pipeline.py --voting-catboost     # LR+SVM+RF+CatBoost
    
    # Specialist ensembles
    python model_pipeline.py --voting-specialist              # Voting + Specialists (α=0.8)
    python model_pipeline.py --voting-specialist --alpha 0.7  # Custom alpha
    python model_pipeline.py --catboost-specialist            # CatBoost + Specialists
    
    # Tune alpha
    python model_pipeline.py --tune-alpha --base voting       # Tune α for Voting
    python model_pipeline.py --tune-alpha --base catboost     # Tune α for CatBoost
    
    # All comparisons
    python model_pipeline.py --all
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Control', 1: 'Breast', 2: 'CRC', 3: 'Gastric', 4: 'Liver', 5: 'Lung'}
N_CLASSES = 6
CRC_IDX = 2
GASTRIC_IDX = 3


def load_data():
    """Load preprocessed data."""
    X = pd.read_parquet(PROCESSED_DIR / 'X_train_final_v3.parquet')
    y = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet')['label'].values
    return X, y


def evaluate_cv(name, model_fn, X, y, cv):
    """Run 5-fold CV for a model function."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    fold_scores = []
    all_preds = np.zeros(len(y), dtype=int)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        y_pred = model_fn(X_tr, y_tr, X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        fold_scores.append(f1)
        all_preds[val_idx] = y_pred
        print(f"  Fold {fold_idx+1}: F1={f1:.3f}")
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    per_class = f1_score(y, all_preds, average=None)
    
    print(f"\n  → F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    
    return {
        'name': name,
        'f1_macro_mean': float(mean_f1),
        'f1_macro_std': float(std_f1),
        'per_class_f1': {LABEL_MAP[i]: float(per_class[i]) for i in range(N_CLASSES)},
        'fold_scores': [float(s) for s in fold_scores]
    }


# ============================================================
# Individual Models
# ============================================================
def run_lr(X_tr, y_tr, X_val):
    model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    return model.predict(X_val)


def run_svm(X_tr, y_tr, X_val):
    model = SVC(C=1.0, gamma='scale', class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    return model.predict(X_val)


def run_rf(X_tr, y_tr, X_val):
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                   random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_tr, y_tr)
    return model.predict(X_val)


def run_xgb(X_tr, y_tr, X_val):
    sample_weights = compute_sample_weight('balanced', y_tr)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
    model.fit(X_tr, y_tr, sample_weight=sample_weights)
    return model.predict(X_val)


def run_catboost_model(X_tr, y_tr, X_val):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.05,
        loss_function='MultiClass', auto_class_weights='Balanced',
        early_stopping_rounds=50, random_state=RANDOM_STATE, verbose=False
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_val).flatten().astype(int)


# ============================================================
# Voting Ensembles
# ============================================================
def run_voting(X, y, cv, use_catboost=False):
    """Voting ensemble (soft voting)."""
    name = "Voting (LR+SVM+RF+CatBoost)" if use_catboost else "Voting (LR+SVM+RF+XGB)"
    print(f"\n{'='*60}")
    print(name)
    print(f"{'='*60}")
    
    if use_catboost:
        from catboost import CatBoostClassifier
    
    fold_scores = []
    all_preds = np.zeros(len(y), dtype=int)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sample_weights = compute_sample_weight('balanced', y_tr)
        
        # Models
        lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
        svm = SVC(C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        
        lr.fit(X_tr, y_tr)
        svm.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        
        if use_catboost:
            cb = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
                                   loss_function='MultiClass', auto_class_weights='Balanced',
                                   random_state=RANDOM_STATE, verbose=False)
            cb.fit(X_tr, y_tr)
            proba = (lr.predict_proba(X_val) + svm.predict_proba(X_val) + 
                    rf.predict_proba(X_val) + cb.predict_proba(X_val)) / 4
        else:
            xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
            xgb.fit(X_tr, y_tr, sample_weight=sample_weights)
            proba = (lr.predict_proba(X_val) + svm.predict_proba(X_val) + 
                    rf.predict_proba(X_val) + xgb.predict_proba(X_val)) / 4
        
        y_pred = np.argmax(proba, axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        fold_scores.append(f1)
        all_preds[val_idx] = y_pred
        print(f"  Fold {fold_idx+1}: F1={f1:.3f}")
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    per_class = f1_score(y, all_preds, average=None)
    print(f"\n  → F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    
    return {
        'name': name,
        'f1_macro_mean': float(mean_f1),
        'f1_macro_std': float(std_f1),
        'per_class_f1': {LABEL_MAP[i]: float(per_class[i]) for i in range(N_CLASSES)},
        'fold_scores': [float(s) for s in fold_scores]
    }


# ============================================================
# Specialist Ensembles
# ============================================================
def run_specialist(X, y, cv, base='voting', alpha=0.8):
    """
    Base model + CRC/Gastric Specialists with probability fusion.
    
    base: 'voting', 'catboost', or 'voting_catboost'
    """
    name = f"{base.title()} + Specialists (α={alpha})"
    print(f"\n{'='*60}")
    print(name)
    print(f"{'='*60}")
    
    if base == 'catboost':
        from catboost import CatBoostClassifier
    
    fold_scores = []
    all_preds = np.zeros(len(y), dtype=int)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sample_weights = compute_sample_weight('balanced', y_tr)
        
        # ========== Base Model ==========
        if base == 'catboost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
                                      loss_function='MultiClass', auto_class_weights='Balanced',
                                      random_state=RANDOM_STATE, verbose=False)
            model.fit(X_tr, y_tr)
            general_proba = model.predict_proba(X_val)
        elif base == 'voting_catboost':
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
            general_proba = (lr.predict_proba(X_val) + svm.predict_proba(X_val) + 
                           rf.predict_proba(X_val) + cb.predict_proba(X_val)) / 4
        else:  # voting
            lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
            svm = SVC(C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
            rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
            xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1)
            
            lr.fit(X_tr, y_tr)
            svm.fit(X_tr, y_tr)
            rf.fit(X_tr, y_tr)
            xgb.fit(X_tr, y_tr, sample_weight=sample_weights)
            
            general_proba = (lr.predict_proba(X_val) + svm.predict_proba(X_val) + 
                           rf.predict_proba(X_val) + xgb.predict_proba(X_val)) / 4
        
        # ========== CRC Specialist ==========
        y_crc = (y_tr == CRC_IDX).astype(int)
        crc_weights = compute_sample_weight('balanced', y_crc)
        crc_model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1,
                                 random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
        crc_model.fit(X_tr, y_crc, sample_weight=crc_weights)
        crc_proba = crc_model.predict_proba(X_val)[:, 1]
        
        # ========== Gastric Specialist ==========
        y_gastric = (y_tr == GASTRIC_IDX).astype(int)
        gastric_weights = compute_sample_weight('balanced', y_gastric)
        gastric_model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1,
                                     random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
        gastric_model.fit(X_tr, y_gastric, sample_weight=gastric_weights)
        gastric_proba = gastric_model.predict_proba(X_val)[:, 1]
        
        # ========== Fusion ==========
        fused = general_proba.copy()
        fused[:, CRC_IDX] = alpha * general_proba[:, CRC_IDX] + (1 - alpha) * crc_proba
        fused[:, GASTRIC_IDX] = alpha * general_proba[:, GASTRIC_IDX] + (1 - alpha) * gastric_proba
        fused = fused / fused.sum(axis=1, keepdims=True)
        
        y_pred = np.argmax(fused, axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        fold_scores.append(f1)
        all_preds[val_idx] = y_pred
        
        crc_f1 = f1_score(y_val == CRC_IDX, y_pred == CRC_IDX)
        gastric_f1 = f1_score(y_val == GASTRIC_IDX, y_pred == GASTRIC_IDX)
        print(f"  Fold {fold_idx+1}: F1={f1:.3f} | CRC={crc_f1:.3f} | Gastric={gastric_f1:.3f}")
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    per_class = f1_score(y, all_preds, average=None)
    print(f"\n  → F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    
    return {
        'name': name,
        'base': base,
        'alpha': alpha,
        'f1_macro_mean': float(mean_f1),
        'f1_macro_std': float(std_f1),
        'per_class_f1': {LABEL_MAP[i]: float(per_class[i]) for i in range(N_CLASSES)},
        'fold_scores': [float(s) for s in fold_scores]
    }


def tune_alpha(X, y, cv, base='voting'):
    """Tune alpha for specialist fusion."""
    print("\n" + "="*60)
    print(f"ALPHA TUNING for {base.title()} + Specialists")
    print("="*60)
    
    alphas = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for alpha in alphas:
        res = run_specialist(X, y, cv, base=base, alpha=alpha)
        results[alpha] = res['f1_macro_mean']
    
    best_alpha = max(results.keys(), key=lambda a: results[a])
    
    print("\n" + "="*60)
    print("ALPHA TUNING RESULTS")
    print("="*60)
    print(f"\n{'α':<8} {'F1 Macro'}")
    print("-"*20)
    for alpha in alphas:
        marker = "★" if alpha == best_alpha else ""
        print(f"{alpha:<8} {results[alpha]:.3f}  {marker}")
    
    print(f"\n🏆 Best α = {best_alpha} (F1={results[best_alpha]:.3f})")
    return best_alpha, results


def print_summary(results):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    sorted_results = sorted(results, key=lambda x: x['f1_macro_mean'], reverse=True)
    
    print(f"\n{'Method':<35} {'F1 Macro':<18}")
    print("-"*55)
    for i, res in enumerate(sorted_results):
        marker = "🏆" if i == 0 else ""
        print(f"{res['name']:<35} {res['f1_macro_mean']:.3f} ± {res['f1_macro_std']:.3f}  {marker}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified cfDNA Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_pipeline.py --lr                        # Logistic Regression
  python model_pipeline.py --catboost                  # CatBoost
  python model_pipeline.py --voting                    # Voting ensemble
  python model_pipeline.py --voting-specialist         # Voting + Specialists
  python model_pipeline.py --catboost-specialist       # CatBoost + Specialists
  python model_pipeline.py --catboost-specialist --alpha 0.7
  python model_pipeline.py --tune-alpha --base catboost
  python model_pipeline.py --all                       # Compare all methods
        """
    )
    
    # Individual models
    parser.add_argument('--lr', action='store_true', help='Logistic Regression')
    parser.add_argument('--svm', action='store_true', help='SVM (RBF)')
    parser.add_argument('--rf', action='store_true', help='Random Forest')
    parser.add_argument('--xgb', action='store_true', help='XGBoost')
    parser.add_argument('--catboost', action='store_true', help='CatBoost')
    
    # Voting
    parser.add_argument('--voting', action='store_true', help='Voting (LR+SVM+RF+XGB)')
    parser.add_argument('--voting-catboost', action='store_true', help='Voting with CatBoost')
    
    # Specialists
    parser.add_argument('--voting-specialist', action='store_true', help='Voting + Specialists')
    parser.add_argument('--catboost-specialist', action='store_true', help='CatBoost + Specialists')
    parser.add_argument('--voting-catboost-specialist', action='store_true', help='Voting(CatBoost) + Specialists')
    parser.add_argument('--alpha', type=float, default=0.8, help='Fusion weight α (default: 0.8)')
    
    # Tuning
    parser.add_argument('--tune-alpha', action='store_true', help='Tune alpha')
    parser.add_argument('--base', type=str, default='voting', choices=['voting', 'catboost'],
                       help='Base model for alpha tuning')
    
    # All
    parser.add_argument('--all', action='store_true', help='Run all methods')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧬 cfDNA Classification Pipeline")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    X, y = load_data()
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    # Individual models
    if args.lr or args.all:
        results.append(evaluate_cv('Logistic Regression', run_lr, X, y, cv))
    if args.svm or args.all:
        results.append(evaluate_cv('SVM (RBF)', run_svm, X, y, cv))
    if args.rf or args.all:
        results.append(evaluate_cv('Random Forest', run_rf, X, y, cv))
    if args.xgb or args.all:
        results.append(evaluate_cv('XGBoost', run_xgb, X, y, cv))
    if args.catboost or args.all:
        results.append(evaluate_cv('CatBoost', run_catboost_model, X, y, cv))
    
    # Voting
    if args.voting or args.all:
        results.append(run_voting(X, y, cv, use_catboost=False))
    if args.voting_catboost:
        results.append(run_voting(X, y, cv, use_catboost=True))
    
    # Specialists
    if args.voting_specialist or args.all:
        results.append(run_specialist(X, y, cv, base='voting', alpha=args.alpha))
    if args.catboost_specialist or args.all:
        results.append(run_specialist(X, y, cv, base='catboost', alpha=args.alpha))
    if args.voting_catboost_specialist:
        results.append(run_specialist(X, y, cv, base='voting_catboost', alpha=args.alpha))
    
    # Alpha tuning
    if args.tune_alpha:
        best_alpha, alpha_results = tune_alpha(X, y, cv, base=args.base)
    
    # Summary
    if len(results) > 1:
        print_summary(results)
    
    # Save
    if args.save and results:
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        path = RESULTS_DIR / 'model_pipeline_results.json'
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved: {path}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
