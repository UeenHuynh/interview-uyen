# PROJECT PLAN v3 - PHASE 2 FEATURE ENGINEERING

**Date**: February 2, 2026  
**Version**: 3.0  
**Author**: Codex  
**Focus**: Feature engineering on raw features (high effort, moderate gain)

---

## Phase 2: Feature Selection Approaches (Option A)

**Goal:** Revisit Stage 2 with a feature engineering + selection stack on **raw features**.

### Option A: Sparse PLS-DA (from plan v3)

Layer A: Sparse PLS-DA -> 50 features (replace PCA)  
Layer B: Stability Selection -> 30 features  
Layer C: Group PCA -> 15 dims

```python
# Layer A: Sparse PLS-DA (feature-level selection)
# - Fit on training data only
# - Select top 50 features by VIP scores

pls = SparsePLSDA(n_components=5, sparsity=0.5, random_state=42)
pls.fit(X_train_scaled, y_train)
vip_scores = compute_vip(pls, X_train_scaled)
top_50_idx = np.argsort(vip_scores)[-50:]

X_train_pls = X_train_scaled[:, top_50_idx]
X_test_pls = X_test_scaled[:, top_50_idx]

# Layer B: Stability selection on top 50 -> 30 features
stable_selector = StabilitySelection(
    base_estimator=LogisticRegression(penalty="l1", solver="saga", max_iter=2000),
    n_resamples=100,
    sample_fraction=0.8,
    threshold=0.6,
    random_state=42,
)
stable_selector.fit(X_train_pls, y_train)
stable_idx = stable_selector.get_support(indices=True)

# Keep top 30 by stability score if more than 30 pass threshold
stable_scores = stable_selector.stability_scores_
top_30_idx = stable_idx[np.argsort(stable_scores[stable_idx])[-30:]]

X_train_stable = X_train_pls[:, top_30_idx]
X_test_stable = X_test_pls[:, top_30_idx]

# Layer C: Group PCA -> 15 dims (per group or combined after mapping)
group_pca = PCA(n_components=15, random_state=42)
X_train_final = group_pca.fit_transform(X_train_stable)
X_test_final = group_pca.transform(X_test_stable)
```

**Notes:**
- Apply all selection/scaling steps using **training data only** to prevent leakage.
- If Group PCA needs per-feature-group handling, keep group indices when selecting top 50.
