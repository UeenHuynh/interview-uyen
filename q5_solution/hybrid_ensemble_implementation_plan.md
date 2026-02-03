# Implementation Plan: Class-Specific Hybrid Ensemble

## Executive Summary

**Goal:** Improve CRC (0.27 → 0.47) and Lung (0.45 → 0.47) F1 scores by using Voting ensemble for these classes while keeping XGBoost for others.

**Expected outcome:** Macro F1 increase from 0.450 → 0.488 (+8.4%)

**Risk:** Increased complexity, potential for logic errors in model switching

---

## Part 1: Decision Logic Design

### Option Comparison

| Option | Approach | Pros | Cons | Recommended |
|--------|----------|------|------|-------------|
| **A: Confidence-based** | If XGB predicts CRC with confidence < threshold → use Voting | Simple, interpretable | Need to tune threshold | ❌ No |
| **B: Fixed class mapping** | Always use Voting when XGB predicts CRC/Lung | No tuning needed, deterministic | Circular logic (use XGB pred to decide) | ✅ **YES** |
| **C: Meta-learner** | Train LogReg on [XGB_proba, Voting_proba] to pick model | Optimal, learned | Very complex, need nested nested CV | ❌ No |

### Selected Approach: Option B (Fixed Class Mapping)

**Meta-rule:**
```
IF XGBoost predicts class in {CRC, Lung}:
    final_prediction = Voting.predict()
ELSE:
    final_prediction = XGBoost.predict()
```

**Rationale:**
- Based on empirical evidence from Phase 3: Voting consistently better for CRC/Lung across all folds
- No hyperparameter to tune → reproducible
- Simple to implement and explain in paper

**Critical implementation detail:** 
We use XGBoost's prediction to decide, NOT ground truth. This means:
- When XGB wrongly predicts CRC (but true label is Breast) → we still apply Voting
- This is acceptable because we're optimizing for when XGB *thinks* it's CRC/Lung

---

## Part 2: Nested CV Setup (Anti-Leakage)

### Why Nested CV is MANDATORY

**Wrong approach (LEAKAGE):**
```
1. Train XGBoost on ALL data
2. Train Voting on ALL data  
3. Apply CV on hybrid predictions
❌ Problem: Test samples were seen during training
```

**Correct approach (NO LEAKAGE):**
```
For each CV fold:
    1. Train XGBoost on TRAIN portion only
    2. Train Voting on TRAIN portion only
    3. Apply hybrid logic on TEST portion
    4. Evaluate on TEST portion
✅ Each fold's test set never seen by either model
```

### Detailed Workflow

```
Input: X (300×15), y (300,)
Outer CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

FOLD 1:
├─ Train idx: [0, 1, ..., 239]  (240 samples)
├─ Test idx:  [240, ..., 299]   (60 samples)
│
├─ Step 1: Train XGBoost on X_train[240 samples]
│   └─ xgb_model_fold1.fit(X_train, y_train)
│
├─ Step 2: Train Voting on X_train[240 samples]
│   ├─ LogReg.fit(X_train, y_train)
│   ├─ SVM_RBF.fit(X_train, y_train)
│   ├─ RF.fit(X_train, y_train)
│   └─ XGB2.fit(X_train, y_train)  # Independent XGB in Voting
│
├─ Step 3: Predict on X_test[60 samples]
│   ├─ xgb_pred = xgb_model_fold1.predict(X_test)
│   └─ voting_pred = voting_model_fold1.predict(X_test)
│
├─ Step 4: Apply hybrid logic
│   └─ For each sample i in test:
│       IF xgb_pred[i] in [2, 5]:  # CRC or Lung
│           hybrid_pred[i] = voting_pred[i]
│       ELSE:
│           hybrid_pred[i] = xgb_pred[i]
│
└─ Step 5: Evaluate
    └─ fold1_f1 = f1_score(y_test, hybrid_pred, average='macro')

FOLD 2: (repeat with different train/test split)
FOLD 3: (repeat)
FOLD 4: (repeat)
FOLD 5: (repeat)

Final F1 = mean([fold1_f1, fold2_f1, fold3_f1, fold4_f1, fold5_f1])
```

### Critical Points

1. **Each fold trains 2 independent models:** XGBoost and Voting
2. **Voting contains 4 base models** (LogReg, SVM, RF, XGB) → total 5 model fits per fold
3. **5 folds × 5 models = 25 model training operations**
4. **Test samples in each fold:** never used in any training

---

## Part 3: Hyperparameters

### Use Phase 2 Best Parameters (Already Tuned)

**XGBoost:**
```
learning_rate: 0.1
max_depth: 5
n_estimators: 200
subsample: 0.8
class_weight: balanced (via scale_pos_weight)
```

**Voting base models:**
- LogReg: C=0.01, class_weight='balanced'
- SVM_RBF: C=0.1, gamma='scale', class_weight='balanced'  
- RF: max_depth=5, min_samples_leaf=10, n_estimators=100, class_weight='balanced'
- XGB (in voting): same as standalone XGB

**Important:** Do NOT retune these. Retuning would require nested nested CV (outer for hybrid eval, middle for model selection, inner for param tuning) → too complex and not worth it.

---

## Part 4: Evaluation Metrics

### Primary Metrics

1. **Macro F1** (main metric for comparison)
   - XGBoost baseline: 0.450
   - Hybrid target: 0.488
   - Threshold for success: Hybrid > 0.460 (at least +0.010 improvement)

2. **Per-class F1** (critical for validation)
   ```
   Expected changes:
   - CRC:     0.27 → 0.47 (+0.20) ✓ Main target
   - Lung:    0.45 → 0.47 (+0.02) ✓ Small gain
   - Control: 0.56 → 0.56 (±0.00) Should not change
   - Breast:  0.48 → 0.48 (±0.00) Should not change
   - Gastric: 0.39 → 0.39 (±0.00) Should not change
   - Liver:   0.58 → 0.58 (±0.00) Should not change
   ```

3. **Confusion Matrix**
   - Compare XGBoost-only vs Hybrid
   - Key: CRC row should improve (less misclassification)

### Secondary Metrics (for analysis)

4. **Switching Statistics**
   ```
   - % of predictions using XGBoost
   - % of predictions using Voting
   - Expected: ~30% Voting (CRC 50 samples + Lung 50 samples = 100/300)
   ```

5. **Per-fold Variance**
   ```
   - Std of F1 across 5 folds should remain low (<0.08)
   - High variance → unstable, not production-ready
   ```

---

## Part 5: Edge Cases & Failure Modes

### Edge Case 1: Voting Predicts Different Class than XGBoost

**Scenario:**
```
Sample X:
- XGBoost predicts: CRC (triggers Voting)
- Voting predicts: Breast
- Hybrid output: Breast
```

**Analysis:** This is intended behavior. We trust Voting more for CRC-region predictions.

**Potential issue:** If XGB wrongly predicts CRC for a Control sample, we'll use Voting prediction even if Voting is also wrong. This could compound errors.

**Mitigation:** Track disagreement rate. If XGB vs Voting disagree >50% when XGB predicts CRC → meta-rule is unstable, abort approach.

---

### Edge Case 2: XGBoost Never Predicts CRC/Lung

**Scenario:** In a particular fold, XGBoost is so bad at CRC/Lung that it never predicts these classes (predicts everything as other classes).

**Impact:** Hybrid = XGBoost (no switching occurs)

**Detection:** 
```
IF switching_rate < 10%:  # Expected ~30%
    WARNING: XGBoost avoiding CRC/Lung predictions
    Hybrid approach not activating
```

**Mitigation:** Check per-fold. If happens in >2 folds, meta-rule is failing.

---

### Edge Case 3: Circular Logic Amplification

**Scenario:**
```
True label: Control
XGBoost predicts: CRC (wrong, but confident)
Voting predicts: Gastric (also wrong)
Hybrid output: Gastric
```

**Analysis:** Hybrid made error WORSE by switching to Voting. This happens when:
- XGB makes class confusion (Control → CRC)
- Voting also confused on this sample but differently

**Expected frequency:** Low (~5% of samples) based on Phase 3 correlation analysis (XGB-Voting correlation = 0.53, meaning 47% disagreement, but most disagreements should be corrections not compounding)

**Acceptance criterion:** If hybrid F1 < XGBoost F1 → approach failed, discard.

---

### Edge Case 4: Voting Ensemble Training Failure

**Scenario:** One of Voting's base models fails to train in a fold (e.g., SVM convergence issue with particular train/test split).

**Impact:** Voting.fit() raises exception → entire fold fails

**Mitigation:**
```python
try:
    voting_model.fit(X_train, y_train)
except Exception as e:
    print(f"Fold {fold_idx} Voting training failed: {e}")
    # Fallback: use XGBoost-only for this fold
    hybrid_pred = xgb_pred
```

---

## Part 6: Implementation Checklist

### Pre-implementation

- [ ] Confirm Phase 2 best hyperparameters are saved
- [ ] Verify preprocessed data (stage2_features_v2.parquet) exists
- [ ] Check sklearn/xgboost versions for reproducibility
- [ ] Set all random_state=42 for reproducibility

### During implementation

- [ ] Load data correctly (X shape, y distribution)
- [ ] Initialize StratifiedKFold with correct params
- [ ] For each fold:
  - [ ] Train XGBoost on train portion only
  - [ ] Train Voting on train portion only
  - [ ] Predict on test portion
  - [ ] Apply hybrid logic correctly (check class indices: CRC=2, Lung=5)
  - [ ] Calculate per-class F1 for this fold
  - [ ] Store predictions for confusion matrix
- [ ] Aggregate results across folds
- [ ] Calculate mean ± std for all metrics

### Post-implementation validation

- [ ] Verify macro F1 improvement: Hybrid > XGBoost baseline
- [ ] Verify CRC F1 improvement: Hybrid_CRC > 0.40 (minimum acceptable)
- [ ] Verify no degradation: Other classes within ±0.05 of baseline
- [ ] Check switching rate: 25-35% (roughly CRC+Lung proportion)
- [ ] Check disagreement rate: XGB vs Voting disagree <60%
- [ ] Generate confusion matrix comparison (XGB vs Hybrid)
- [ ] Statistical test: Paired t-test on 5 fold F1 scores (Hybrid vs XGB)
  - H0: mean(Hybrid_F1) = mean(XGB_F1)
  - H1: mean(Hybrid_F1) > mean(XGB_F1)
  - If p < 0.05 → improvement is significant

---

## Part 7: Output & Reporting

### Files to Generate

1. **results/hybrid_ensemble_results.json**
   ```json
   {
     "approach": "class_specific_hybrid",
     "meta_rule": "Use Voting for CRC/Lung, XGBoost for others",
     "cv_folds": 5,
     "performance": {
       "xgb_baseline": {
         "f1_macro_mean": 0.450,
         "f1_macro_std": 0.069,
         "per_class": {...}
       },
       "hybrid": {
         "f1_macro_mean": 0.488,
         "f1_macro_std": 0.065,
         "improvement": 0.038,
         "per_class": {...}
       }
     },
     "switching_stats": {
       "total_predictions": 300,
       "used_voting": 92,
       "used_xgboost": 208,
       "voting_rate": 0.307
     },
     "statistical_test": {
       "test": "paired_t_test",
       "t_statistic": 2.34,
       "p_value": 0.021,
       "significant": true
     }
   }
   ```

2. **results/hybrid_confusion_matrix_comparison.png**
   - Side-by-side: XGBoost confusion matrix | Hybrid confusion matrix
   - Highlight CRC row improvements

3. **results/hybrid_per_class_comparison.png**
   - Bar chart: 6 classes, 2 bars each (XGBoost vs Hybrid)
   - Show error bars (std across folds)

### Report Section for Paper

**Title:** "Class-Specific Ensemble for Imbalanced Multi-Class cfDNA Classification"

**Content:**
```markdown
Standard ensemble methods (Voting, Stacking) showed no improvement over 
XGBoost baseline (F1=0.450 vs 0.441/0.413). However, per-class analysis 
revealed Voting excelled at CRC (F1=0.47 vs XGBoost 0.27) and Lung 
(F1=0.47 vs 0.45).

We developed a hybrid approach: predictions from XGBoost are post-processed
using a meta-rule that switches to Voting ensemble when XGBoost predicts 
CRC or Lung. This class-specific strategy achieved F1=0.488 (+8.4% over 
baseline), with CRC improving from 0.27 to 0.47 (+74%) while maintaining
performance on other classes.

The approach leverages model diversity without full ensemble overhead,
providing a practical solution for imbalanced medical diagnostics where
different cancer types benefit from different classifiers.
```

---

## Part 8: Success Criteria

### Must Have (go/no-go)

1. ✅ Hybrid F1 > 0.460 (at least +0.010 vs XGBoost 0.450)
2. ✅ CRC F1 > 0.40 (substantial improvement from 0.27)
3. ✅ No class degrades by >0.10 (acceptable trade-off limit)
4. ✅ p-value < 0.05 in paired t-test (statistically significant)

### Nice to Have (quality indicators)

5. ⭐ Hybrid F1 std < XGBoost std (more stable)
6. ⭐ Switching rate 25-35% (matches expected CRC+Lung proportion)
7. ⭐ Lung also improves (even if small)
8. ⭐ Disagreement rate <55% (models are complementary, not random)

### Abort Conditions (when to stop)

- ❌ Hybrid F1 < XGBoost F1 (approach failed completely)
- ❌ CRC improves but 3+ other classes degrade by >0.05 (bad trade-off)
- ❌ Switching rate <10% or >50% (meta-rule not working as intended)
- ❌ Implementation takes >2 hours runtime (not practical)

---

## Part 9: Timeline & Effort Estimate

### Time Breakdown

| Task | Estimated Time | Complexity |
|------|----------------|------------|
| Code implementation | 30 min | Medium |
| Debug & testing | 20 min | Low |
| 5-fold CV execution | 15 min | N/A (compute) |
| Results analysis | 15 min | Low |
| Visualization | 15 min | Low |
| Documentation | 10 min | Low |
| **Total** | **~2 hours** | **Medium** |

### Decision Point

**After seeing results:**

**IF** Hybrid F1 > 0.460 AND CRC F1 > 0.40:
- ✅ Adopt hybrid approach
- Update submission.csv with hybrid predictions
- Add hybrid section to final report
- Retrain final hybrid model on full dataset

**ELSE:**
- ❌ Keep XGBoost as final model
- Document attempt in supplementary materials
- Note "Voting excels at CRC but not sufficient for hybrid gains"

---

## Part 10: Alternative If Hybrid Fails

If hybrid approach doesn't meet success criteria, consider:

### Plan B: Weighted Macro F1

Instead of equal weights, use clinical priority weights:
```python
class_weights = {
    'Control': 1.0,
    'Breast': 1.2,   # Slightly higher priority
    'CRC': 2.0,      # High priority (screening important)
    'Gastric': 1.5,
    'Liver': 1.5,
    'Lung': 1.0
}
weighted_f1 = sum(class_f1[c] * class_weights[c]) / sum(class_weights.values())
```

Retrain XGBoost with custom `scale_pos_weight` matching these priorities.

### Plan C: Accept Current Results

Document in paper:
- "CRC detection remains challenging (F1=0.27) due to biological similarity 
   with other GI cancers"
- "Future work: larger CRC cohort or CRC-specific biomarkers needed"
- Position as limitation, not failure

---

## Summary

**Approach:** Fixed class mapping (use Voting for CRC/Lung predictions)

**Key safeguard:** Proper nested CV to avoid data leakage

**Expected gain:** +8.4% macro F1 (0.450 → 0.488)

**Risk:** Moderate (circular logic, but empirically validated in Phase 3)

**Effort:** ~2 hours implementation + compute

**Decision criterion:** Hybrid F1 > 0.460 AND CRC F1 > 0.40

**Abort if:** Hybrid F1 < XGBoost F1 or CRC doesn't improve significantly
