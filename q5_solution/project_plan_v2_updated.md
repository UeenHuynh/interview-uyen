# UPDATED PROJECT PLAN - CANCER CLASSIFICATION
## Aligned with Data Discovery Insights

**Date Updated**: January 31, 2026  
**Version**: 2.0 (Aligned with Data Discovery)

---

## ⚠️ CRITICAL DATA INSIGHTS (From Discovery)

Based on comprehensive data discovery, the following CRITICAL issues must be addressed:

### 🔴 SEVERITY: HIGH

1. **Extreme Feature Redundancy**
   - NUCLEOSOME: 5,403 highly correlated pairs (>0.95) - **90% of possible pairs!**
   - FLEN: 1,463 correlated pairs
   - EM: 161 correlated pairs
   - **Impact**: Severe multicollinearity, model instability
   - **Action**: MANDATORY aggressive correlation filtering or PCA

2. **High Dimensionality Problem**
   - Features: 1,158
   - Samples: 300
   - Ratio: 3.86:1 (features >> samples)
   - **Impact**: Extreme overfitting risk, curse of dimensionality
   - **Action**: MUST reduce to max 100-150 features

### ⚠️ SEVERITY: MEDIUM

3. **Liver Class Severely Underrepresented**
   - Liver: 30 samples (10%) vs others: 50-70 samples
   - **Impact**: Poor Liver classification expected
   - **Action**: Specific handling needed (SMOTE, class weights, or accept lower performance)

4. **Class Imbalance (Overall)**
   - Control: 70 (23.3%)
   - Cancer types: 30-50 each
   - **Impact**: Potential bias toward Control class
   - **Action**: Stratified sampling, balanced metrics

### ✅ STRENGTHS

- ✅ No missing values
- ✅ Clean, consistent data
- ✅ Multiple complementary feature views
- ✅ Well-labeled samples

---

## REVISED IMPLEMENTATION ROADMAP

### PHASE 0: MANDATORY PREPROCESSING (UPDATED)

This phase is now **CRITICAL** given data characteristics:

#### Step 0.1: Data Preparation (30 min)
```python
# Transpose datasets
X_em = em_df.T          # (300, 256)
X_flen = flen_df.T      # (300, 301)
X_nuc = nucleosome_df.T # (300, 601)

# Extract labels
y = extract_labels(em_df.columns)
# Encoding: 0=Control, 1=Breast, 2=CRC, 3=Gastric, 4=Liver, 5=Lung

# Stratified split (MANDATORY given class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### Step 0.2: **CRITICAL** - Correlation Filtering (1 hour)

**PRIORITY: Handle extreme correlation before anything else**

```python
def remove_correlated_features(X, threshold=0.95):
    """
    NUCLEOSOME has 5,403 correlated pairs!
    Must remove before proceeding.
    """
    corr_matrix = np.corrcoef(X.T)
    upper = np.triu(corr_matrix, k=1)
    to_drop = []
    
    for i in range(len(upper)):
        if any(abs(upper[i, :]) > threshold):
            to_drop.append(i)
    
    # Keep features not in to_drop
    keep = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep], keep

# Apply to each feature set SEPARATELY
X_em_filtered, em_keep = remove_correlated_features(X_em, 0.95)
X_flen_filtered, flen_keep = remove_correlated_features(X_flen, 0.95)
X_nuc_filtered, nuc_keep = remove_correlated_features(X_nuc, 0.95)

print(f"EM: {len(em_keep)}/{256} features kept")
print(f"FLEN: {len(flen_keep)}/{301} features kept")
print(f"NUCLEOSOME: {len(nuc_keep)}/{601} features kept")

# Expected reduction: ~50-70% of features removed
```

#### Step 0.3: Feature Scaling (15 min)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)
```

---

## PHASE 1: EXPLORATORY DATA ANALYSIS (2-3 hours)

### 1.1 Individual Feature Set Analysis (NEW - CRITICAL)

**Test each feature set separately BEFORE combining:**

```python
# Baseline performance for each feature type
for name, X in [('EM', X_em_filtered), 
                ('FLEN', X_flen_filtered), 
                ('NUCLEOSOME', X_nuc_filtered)]:
    
    # Quick Random Forest test
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')
    print(f"{name}: F1={scores.mean():.3f} ± {scores.std():.3f}")
```

**Purpose:** 
- Identify which feature set is most informative
- Decide if all three are needed or if one/two dominate
- Guide feature fusion strategy

### 1.2 PCA Visualization (45 min)

**Given extreme correlation, PCA will be very effective:**

```python
from sklearn.decomposition import PCA

# PCA on combined features (after correlation filtering)
pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)

# Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
plt.axhline(y=0.99, color='g', linestyle='--', label='99%')

# How many components for 95%? Likely << 100
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Visualize first 2 PCs
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='tab10')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Class')
```

### 1.3 Class Distribution Visualization (30 min)

```python
import seaborn as sns

# Class counts
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training set
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title('Training Set Distribution')
axes[0].set_xticklabels(['Control', 'Breast', 'CRC', 'Gastric', 'Liver', 'Lung'])

# Test set
sns.countplot(x=y_test, ax=axes[1])
axes[1].set_title('Test Set Distribution')
axes[1].set_xticklabels(['Control', 'Breast', 'CRC', 'Gastric', 'Liver', 'Lung'])

plt.tight_layout()
```

### 1.4 t-SNE Visualization (30 min)

```python
from sklearn.manifold import TSNE

# Use PCA-reduced data for speed
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca[:, :50])  # Use first 50 PCs

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Class')
plt.title('t-SNE Visualization')
```

---

## PHASE 2: FEATURE SELECTION & REDUCTION (3-4 hours)

### UPDATED Priority Order:

1. **Correlation Filtering** (DONE in Phase 0) ✓
2. **PCA Reduction** (MANDATORY given data)
3. **Statistical Selection** (ANOVA F-test)
4. **Tree-based Importance** (Random Forest)

### 2.1 PCA-Based Dimensionality Reduction (1 hour)

**RECOMMENDED PRIMARY APPROACH** given extreme correlation:

```python
from sklearn.decomposition import PCA

# Method A: Keep 95% variance
pca_95 = PCA(n_components=0.95, random_state=42)
X_train_pca95 = pca_95.fit_transform(X_train_scaled)
X_test_pca95 = pca_95.transform(X_test_scaled)

print(f"Original features: {X_train_scaled.shape[1]}")
print(f"PCA components (95%): {X_train_pca95.shape[1]}")

# Method B: Keep 99% variance
pca_99 = PCA(n_components=0.99, random_state=42)
X_train_pca99 = pca_99.fit_transform(X_train_scaled)
X_test_pca99 = pca_99.transform(X_test_scaled)

print(f"PCA components (99%): {X_train_pca99.shape[1]}")

# Method C: Fixed number (e.g., 50-100)
pca_100 = PCA(n_components=100, random_state=42)
X_train_pca100 = pca_100.fit_transform(X_train_scaled)
X_test_pca100 = pca_100.transform(X_test_scaled)

# Compare all three in baseline models
```

### 2.2 Statistical Feature Selection (1 hour)

**Alternative to PCA - select original features:**

```python
from sklearn.feature_selection import SelectKBest, f_classif

# ANOVA F-test
selector = SelectKBest(f_classif, k=100)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
print(f"Top 100 features by ANOVA F-test selected")

# Analyze which feature sets contribute most
em_selected = sum(i < 256 for i in selected_indices)
flen_selected = sum(256 <= i < 557 for i in selected_indices)
nuc_selected = sum(i >= 557 for i in selected_indices)

print(f"EM: {em_selected}, FLEN: {flen_selected}, NUCLEOSOME: {nuc_selected}")
```

### 2.3 Tree-Based Feature Importance (1 hour)

```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest on all features
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf.feature_importances_

# Select top 100
top_100_idx = np.argsort(importances)[-100:]
X_train_rf = X_train_scaled[:, top_100_idx]
X_test_rf = X_test_scaled[:, top_100_idx]
```

### 2.4 Comparison of Methods

**Test which dimensionality reduction works best:**

```python
results = {}

for name, X_tr, X_te in [
    ('PCA_95', X_train_pca95, X_test_pca95),
    ('PCA_99', X_train_pca99, X_test_pca99),
    ('PCA_100', X_train_pca100, X_test_pca100),
    ('ANOVA_100', X_train_selected, X_test_selected),
    ('RF_100', X_train_rf, X_test_rf)
]:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_tr, y_train, cv=5, scoring='f1_macro')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: F1={scores.mean():.3f} ± {scores.std():.3f}")

# Choose best method for next phases
best_method = max(results, key=lambda k: results[k]['mean'])
print(f"\nBest method: {best_method}")
```

---

## PHASE 3: BASELINE MODELING (2-3 hours)

### 3.1 Baseline Models with Best Feature Set

Use the best dimensionality reduction method from Phase 2:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

baseline_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_best, y_train, 
                            cv=StratifiedKFold(5), 
                            scoring='f1_macro')
    baseline_results[name] = scores.mean()
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 3.2 Handle Liver Class Imbalance (NEW)

**Specific strategies for underrepresented Liver class:**

#### Option A: Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute balanced class weights
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Use in models
rf_weighted = RandomForestClassifier(
    n_estimators=100, 
    class_weight=class_weight_dict,
    random_state=42
)
```

#### Option B: SMOTE Oversampling
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_best, y_train)

print(f"Before SMOTE: {Counter(y_train)}")
print(f"After SMOTE: {Counter(y_train_balanced)}")

# Train on balanced data
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_balanced, y_train_balanced)
```

#### Option C: Stratified Cross-Validation with Monitoring
```python
# Monitor per-class performance specifically
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_best, y_train)):
    X_fold_train, X_fold_val = X_train_best[train_idx], X_train_best[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    y_pred = model.predict(X_fold_val)
    
    # Per-class F1
    f1_per_class = f1_score(y_fold_val, y_pred, average=None)
    print(f"Fold {fold+1} - Liver F1: {f1_per_class[4]:.3f}")  # Class 4 is Liver
```

---

## PHASE 4: STRATEGY IMPLEMENTATION (4-6 hours)

### Strategy 1: Two-Stage Classification

#### Stage 1: Control vs Disease (Binary)

```python
# Create binary labels
y_binary = (y_train > 0).astype(int)  # 0=Control, 1=Disease

# Test multiple models
models_stage1 = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200),
    'XGBoost': XGBClassifier(n_estimators=200),
    'SVM': SVC(kernel='rbf', probability=True)
}

stage1_results = {}
for name, model in models_stage1.items():
    scores = cross_val_score(model, X_train_best, y_binary, 
                            cv=5, scoring='roc_auc')
    stage1_results[name] = scores.mean()
    print(f"{name}: AUC={scores.mean():.3f}")

# Select best model for Stage 1
best_stage1 = max(stage1_results, key=stage1_results.get)
```

#### Stage 2: Cancer Type Classification (5-class)

```python
# Filter only disease samples
disease_mask = y_train > 0
X_train_disease = X_train_best[disease_mask]
y_train_disease = y_train[disease_mask] - 1  # Remap to 0-4

# Train Stage 2 model
models_stage2 = {
    'Random Forest': RandomForestClassifier(n_estimators=200),
    'XGBoost': XGBClassifier(n_estimators=200),
    'SVM': SVC(kernel='rbf')
}

stage2_results = {}
for name, model in models_stage2.items():
    scores = cross_val_score(model, X_train_disease, y_train_disease,
                            cv=5, scoring='f1_macro')
    stage2_results[name] = scores.mean()
    print(f"{name}: F1={scores.mean():.3f}")
```

#### Cascade Evaluation

```python
# Simulate full pipeline
def evaluate_cascade(stage1_model, stage2_model, X_test, y_test):
    # Stage 1 predictions
    y_binary_pred = stage1_model.predict(X_test)
    
    # Initialize final predictions
    y_final_pred = np.zeros(len(y_test))
    
    # Stage 2 for predicted disease
    disease_idx = y_binary_pred == 1
    if disease_idx.sum() > 0:
        X_disease = X_test[disease_idx]
        y_disease_pred = stage2_model.predict(X_disease) + 1
        y_final_pred[disease_idx] = y_disease_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_final_pred)
    f1_macro = f1_score(y_test, y_final_pred, average='macro')
    
    return accuracy, f1_macro, y_final_pred

# Train final models
final_stage1 = models_stage1[best_stage1]
final_stage1.fit(X_train_best, y_binary)

final_stage2 = models_stage2[max(stage2_results, key=stage2_results.get)]
final_stage2.fit(X_train_disease, y_train_disease)

# Evaluate
acc, f1, preds = evaluate_cascade(final_stage1, final_stage2, X_test_best, y_test)
print(f"Strategy 1 - Accuracy: {acc:.3f}, F1: {f1:.3f}")
```

### Strategy 2: Direct 6-Class Classification

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models_6class = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=1,  # Will tune per-class
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42
    )
}

strategy2_results = {}
for name, model in models_6class.items():
    scores = cross_val_score(model, X_train_best, y_train,
                            cv=StratifiedKFold(5),
                            scoring='f1_macro')
    strategy2_results[name] = scores.mean()
    print(f"{name}: F1={scores.mean():.3f} ± {scores.std():.3f}")

# Best model for Strategy 2
best_strategy2_name = max(strategy2_results, key=strategy2_results.get)
best_strategy2_model = models_6class[best_strategy2_name]
```

---

## PHASE 5: HYPERPARAMETER TUNING (3-4 hours)

### 5.1 Random Forest Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=StratifiedKFold(5),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

rf_random.fit(X_train_best, y_train)
print(f"Best params: {rf_random.best_params_}")
print(f"Best score: {rf_random.best_score_:.3f}")
```

### 5.2 XGBoost Tuning

```python
param_dist_xgb = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions=param_dist_xgb,
    n_iter=50,
    cv=StratifiedKFold(5),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

xgb_random.fit(X_train_best, y_train)
```

---

## PHASE 6: FINAL EVALUATION (2-3 hours)

### 6.1 Compare Both Strategies

```python
# Strategy 1 (Two-stage)
strategy1_pred = evaluate_cascade(final_stage1, final_stage2, X_test_best, y_test)

# Strategy 2 (Direct 6-class)
best_strategy2_model.fit(X_train_best, y_train)
strategy2_pred = best_strategy2_model.predict(X_test_best)

# Compare
print("STRATEGY COMPARISON:")
print(f"Strategy 1 (Two-stage): F1={f1_score(y_test, strategy1_pred[2], average='macro'):.3f}")
print(f"Strategy 2 (6-class):   F1={f1_score(y_test, strategy2_pred, average='macro'):.3f}")

# Detailed per-class metrics
print("\nPer-class F1-scores:")
print("Class        Strategy1  Strategy2")
class_names = ['Control', 'Breast', 'CRC', 'Gastric', 'Liver', 'Lung']
f1_s1 = f1_score(y_test, strategy1_pred[2], average=None)
f1_s2 = f1_score(y_test, strategy2_pred, average=None)

for i, name in enumerate(class_names):
    print(f"{name:12s} {f1_s1[i]:.3f}      {f1_s2[i]:.3f}")
```

### 6.2 Confusion Matrices

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Strategy 1
cm1 = confusion_matrix(y_test, strategy1_pred[2])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Strategy 1: Two-Stage')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Strategy 2
cm2 = confusion_matrix(y_test, strategy2_pred)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Strategy 2: Direct 6-Class')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
```

### 6.3 Feature Importance Analysis

```python
# For tree-based models
if hasattr(best_strategy2_model, 'feature_importances_'):
    importances = best_strategy2_model.feature_importances_
    
    # If using PCA
    if 'PCA' in best_method:
        # Show PC importances
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.xlabel('Principal Component')
        plt.ylabel('Importance')
        plt.title('PC Importance')
    else:
        # Show top 20 original features
        top_20_idx = np.argsort(importances)[-20:]
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importances[top_20_idx])
        plt.ylabel('Feature')
        plt.xlabel('Importance')
        plt.title('Top 20 Features')
```

---

## UPDATED TIMELINE ESTIMATE

Given the complexity revealed by data discovery:

| Phase | Task | Original | Updated | Reason |
|-------|------|----------|---------|--------|
| 0 | **Mandatory Preprocessing** | - | **1-2 hours** | NEW: Correlation filtering critical |
| 1 | EDA & Visualization | 2-3 hours | **2-3 hours** | Same |
| 2 | Feature Selection | 1-2 hours | **3-4 hours** | More complex due to correlation |
| 3 | Baseline Models | 1-2 hours | **2-3 hours** | Added Liver imbalance handling |
| 4 | Strategy Implementation | 2-3 hours | **4-6 hours** | More thorough given challenges |
| 5 | Hyperparameter Tuning | 3-4 hours | **3-4 hours** | Same |
| 6 | Final Evaluation | 1-2 hours | **2-3 hours** | More detailed comparison needed |
| 7 | Documentation | 2-3 hours | **2-3 hours** | Same |

**Original Total**: ~15-20 hours  
**Updated Total**: **~19-28 hours** (more realistic given data complexity)

---

## CRITICAL SUCCESS FACTORS (UPDATED)

### Top Priorities (Ordered by Importance):

1. **🔴 MUST: Aggressive dimensionality reduction**
   - Cannot proceed with 1,158 features
   - PCA or correlation filtering mandatory
   - Target: 50-150 features max

2. **🔴 MUST: Handle extreme correlation**
   - NUCLEOSOME correlation is critical issue
   - Will cause severe multicollinearity
   - Remove or use PCA

3. **⚠️ SHOULD: Address Liver class imbalance**
   - Only 30 samples vs 50-70 others
   - Use class weights or SMOTE
   - Accept lower Liver performance as baseline

4. **⚠️ SHOULD: Use stratified sampling everywhere**
   - Train/test split
   - Cross-validation
   - All evaluation

5. **✅ NICE: Test feature sets independently**
   - Understand which is most informative
   - May guide fusion strategy

---

## EXPECTED OUTCOMES (REVISED)

### Realistic Performance Targets:

Given the challenges:

| Metric | Conservative | Realistic | Optimistic |
|--------|-------------|-----------|------------|
| Overall Accuracy | 65-70% | 75-80% | 85-90% |
| Macro F1-score | 0.60-0.65 | 0.70-0.75 | 0.80-0.85 |
| Control F1 | 0.75-0.80 | 0.85-0.90 | 0.90-0.95 |
| Cancer avg F1 | 0.55-0.60 | 0.65-0.70 | 0.75-0.80 |
| **Liver F1** | **0.40-0.50** | **0.55-0.65** | **0.70-0.75** |

**Key Notes:**
- Liver performance will likely be lowest
- Control vs Disease easier than cancer type discrimination
- Strategy 1 may outperform Strategy 2 due to divide-and-conquer
- Feature selection quality will heavily impact results

---

## DELIVERABLES CHECKLIST

### Code & Notebooks
- [ ] Complete Jupyter notebook with all analysis
- [ ] Modular Python functions for reusability
- [ ] Saved preprocessing pipeline (scaler, selector)
- [ ] Saved final models (pickle/joblib)
- [ ] Requirements.txt with all dependencies

### Visualizations
- [ ] PCA/t-SNE plots (colored by class)
- [ ] Correlation heatmaps (before/after filtering)
- [ ] Feature importance plots
- [ ] Confusion matrices (both strategies)
- [ ] ROC curves (if applicable)
- [ ] Learning curves
- [ ] Per-class performance comparison

### Documentation
- [ ] Executive summary
- [ ] Methodology description
- [ ] Data preprocessing steps
- [ ] Feature selection rationale
- [ ] Model comparison table
- [ ] Best hyperparameters
- [ ] Performance metrics
- [ ] Limitations and caveats
- [ ] Recommendations (Strategy 1 vs 2)
- [ ] Future improvements

### Results Summary
- [ ] Strategy comparison table
- [ ] Per-class performance breakdown
- [ ] Confusion matrices analysis
- [ ] Feature importance interpretation
- [ ] Clinical relevance discussion

---

## FINAL RECOMMENDATIONS

### Primary Recommendation:

**Use PCA-based dimensionality reduction as PRIMARY approach**

Rationale:
- Handles extreme correlation automatically
- Proven effective for genomic data
- Reduces from 1,158 → ~50-100 components
- Orthogonal features (no multicollinearity)
- Captures 95-99% variance

### Secondary Recommendation:

**Start with Strategy 2 (Direct 6-class), fall back to Strategy 1 if needed**

Rationale:
- Simpler conceptually
- No cascade error
- Easier to optimize
- Strategy 1 as backup if poor cancer discrimination

### Tertiary Recommendation:

**Use ensemble of XGBoost + Random Forest for final model**

Rationale:
- Both handle high dimensions well
- XGBoost excellent for imbalanced data
- Random Forest provides robustness
- Ensemble reduces overfitting

### Handling Liver Class:

**Use class weights, monitor separately, set realistic expectations**

Approach:
- Apply balanced class weights
- Track Liver F1 separately
- Accept 0.55-0.65 F1 as success
- Don't oversample (may cause overfitting with only 30 samples)

---

## APPENDIX: UPDATED CODE SNIPPETS

### Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 1. Load data
em_df = pd.read_csv('EM.csv', index_col=0)
flen_df = pd.read_csv('FLEN.csv', index_col=0)
nuc_df = pd.read_csv('NUCLEOSOME.csv', index_col=0)

# 2. Transpose
X_em = em_df.T.values
X_flen = flen_df.T.values
X_nuc = nuc_df.T.values

# 3. Extract labels
def extract_labels(columns):
    labels = []
    label_map = {'Control': 0, 'Breast': 1, 'CRC': 2, 
                 'Gastric': 3, 'Liver': 4, 'Lung': 5}
    for col in columns:
        for key in label_map:
            if key in col:
                labels.append(label_map[key])
                break
    return np.array(labels)

y = extract_labels(em_df.columns)

# 4. Combine features
X = np.hstack([X_em, X_flen, X_nuc])
print(f"Combined shape: {X.shape}")  # (300, 1158)

# 5. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. PCA dimensionality reduction
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA reduced to: {X_train_pca.shape[1]} components")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
```

---

**END OF UPDATED PROJECT PLAN**

This plan is now fully aligned with the critical insights from data discovery and provides realistic, actionable steps for successful model development.
