# DATA DISCOVERY REPORT
## Cancer Classification from cfDNA Features

**Date**: January 31, 2026  
**Analyst**: Claude  
**Project**: Multi-class Cancer Classification Using Cell-Free DNA Features

---

## EXECUTIVE SUMMARY

This report presents a comprehensive data discovery analysis for a cancer classification task using cell-free DNA (cfDNA) features. The dataset comprises 300 samples across 6 classes (1 healthy control + 5 cancer types) with three distinct feature sets: End Motif patterns, Fragment Length distributions, and Nucleosome positioning signals.

**Key Findings:**
- ✅ Clean dataset with no missing values
- ⚠️ Moderate class imbalance (Liver cancer underrepresented at 10%)
- ⚠️ High feature dimensionality (1,158 total features)
- ⚠️ Significant feature redundancy detected (high correlations)
- ✅ All features are continuous numerical values
- ⚠️ Features require scaling due to different value ranges

---

## 1. DATASET OVERVIEW

### 1.1 Data Sources

We have three CSV files containing different types of cfDNA features:

| Dataset | File | Description | Purpose |
|---------|------|-------------|---------|
| **EM** | EM.csv | End Motif patterns | Captures DNA fragmentation patterns at fragment ends |
| **FLEN** | FLEN.csv | Fragment Length distribution | Represents size distribution of cfDNA fragments |
| **NUCLEOSOME** | NUCLEOSOME.csv | Nucleosome positioning | Reflects chromatin structure and nucleosome occupancy |

### 1.2 Data Structure

**Current Format:**
- Data is stored in **transposed** format
- Rows = Features (e.g., "CCCA", "50.0", "-300")
- Columns = Samples (e.g., "Control_1", "Breast_2", "Lung_15")

**Required Transformation:**
- Need to transpose for ML models
- Target format: Rows = Samples, Columns = Features
- Labels will be extracted from sample names

### 1.3 Dimensions Summary

| Dataset | Features | Samples | Missing Values | Data Type |
|---------|----------|---------|----------------|-----------|
| **EM** | 256 | 300 | 0 | float64 |
| **FLEN** | 301 | 300 | 0 | float64 |
| **NUCLEOSOME** | 601 | 300 | 0 | float64 |
| **TOTAL** | **1,158** | **300** | **0** | **float64** |

**Observations:**
- ✅ No missing values - excellent data quality
- ✅ Consistent number of samples across all datasets
- ⚠️ High dimensionality: 1,158 features vs 300 samples (ratio 3.86:1)
- ⚠️ Risk of overfitting due to high feature-to-sample ratio
- ✅ All features are continuous numerical values

---

## 2. CLASS DISTRIBUTION ANALYSIS

### 2.1 Sample Distribution

| Class | Count | Percentage | Category |
|-------|-------|------------|----------|
| **Control** | 70 | 23.3% | Healthy |
| **Breast** | 50 | 16.7% | Cancer |
| **CRC** (Colorectal) | 50 | 16.7% | Cancer |
| **Gastric** | 50 | 16.7% | Cancer |
| **Liver** | 30 | 10.0% | Cancer |
| **Lung** | 50 | 16.7% | Cancer |
| **TOTAL** | **300** | **100%** | - |

### 2.2 Class Balance Analysis

```
Class Distribution Visualization:
Control ████████████████████████ 70
Breast  ████████████████         50
CRC     ████████████████         50
Gastric ████████████████         50
Lung    ████████████████         50
Liver   ██████████               30
```

**Imbalance Ratio:** 
- Max/Min ratio: 70/30 = 2.33:1
- Control vs Disease: 70/230 = 0.30 (30% healthy, 70% disease)

**Implications:**
- ⚠️ **Moderate class imbalance** - Liver cancer is underrepresented
- ⚠️ Need to use **stratified sampling** in train/test split
- ⚠️ Consider **class weights** or **SMOTE** for balancing
- ✅ Imbalance is manageable (not severe like 1:10 or worse)
- 📊 For Strategy 1: Binary classification (Control vs Disease) is relatively balanced

**Recommendations:**
1. Use `StratifiedKFold` for cross-validation
2. Use `stratify` parameter in `train_test_split`
3. Report per-class metrics (Precision, Recall, F1) not just overall accuracy
4. Consider weighted F1-score as primary metric
5. For Strategy 2 (6-class), may need to oversample Liver or undersample others

---

## 3. FEATURE ANALYSIS

### 3.1 End Motif (EM) Features

**Feature Type:** DNA sequence patterns at fragment ends  
**Number of Features:** 256  
**Feature Names:** 4-nucleotide combinations (e.g., "CCCA", "CCAG", "CCTG")

**Examples of EM Features:**
```
1. CCCA    5. CCAA    9. TGAA
2. CCAG    6. CCCT   10. CCTC
3. CCTG    7. CCAT   11. CCGA
4. CAAA    8. CCTT   12. ACAA
```

**Statistical Summary:**
| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 0.003906 | Each motif appears ~0.4% on average |
| Std Dev | 0.003380 | Relatively consistent variability |
| Min | 0.000054 | Rarest motifs |
| Max | 0.027595 | Most common motifs |
| Range | [0.000054, 0.027595] | Wide range, need scaling |

**Key Observations:**
- ✅ All values are **proportions** (frequencies)
- ✅ Sum of all EM features per sample ≈ 1.0
- 📊 Wide range suggests some motifs are much more common
- ⚠️ High correlation detected (161 feature pairs with r > 0.95)
- ⚠️ 253 features have extremely low variance (< 1e-6)

**Biological Interpretation:**
- End motifs reflect DNA fragmentation preferences
- Different tissues may have different fragmentation patterns
- Cancer may alter DNA fragmentation machinery

### 3.2 Fragment Length (FLEN) Features

**Feature Type:** Size distribution of cfDNA fragments  
**Number of Features:** 301  
**Feature Names:** Fragment lengths from 50 to 350 base pairs

**Fragment Length Range:**
```
50bp, 51bp, 52bp, ..., 349bp, 350bp
```

**Statistical Summary:**
| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 0.003322 | Each length ~0.3% on average |
| Std Dev | 0.006337 | High variability |
| Min | 0.000000 | Some lengths absent in samples |
| Max | 0.041626 | Peak frequencies |
| Range | [0.0, 0.041626] | Contains zeros, need careful handling |

**Key Observations:**
- ✅ Values are **frequency distributions**
- ⚠️ Contains **zeros** - some fragment lengths don't appear
- ⚠️ High correlation (1,463 pairs with r > 0.95)
- ⚠️ 227 features with very low variance
- 📊 Likely has peak around 167bp (nucleosome-sized fragments)

**Biological Interpretation:**
- cfDNA shows characteristic fragment length patterns
- Mono-nucleosomal (~167bp) and di-nucleosomal (~334bp) peaks
- Cancer may alter fragment size distribution
- Different cancers may have distinct fragmentation profiles

### 3.3 Nucleosome Positioning Features

**Feature Type:** Coverage patterns around transcription start sites (TSS)  
**Number of Features:** 601  
**Feature Names:** Positions from -300 to +300 bp relative to TSS

**Position Range:**
```
-300, -299, -298, ..., -1, 0, +1, ..., +299, +300
```

**Statistical Summary:**
| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 0.001664 | Average coverage at each position |
| Std Dev | 0.001949 | Moderate variability |
| Min | 0.000063 | Minimum coverage |
| Max | 0.006160 | Maximum coverage |
| Range | [0.000063, 0.006160] | Relatively narrow |

**Key Observations:**
- ✅ Represents **coverage signal** around genes
- ⚠️ **Extremely high correlation** (5,403 pairs with r > 0.95)
- ⚠️ **All 601 features** have low variance
- 📊 Sequential positions are highly correlated (expected)
- 🔍 May show nucleosome depletion near TSS

**Biological Interpretation:**
- Nucleosome positioning reflects chromatin accessibility
- TSS regions typically show nucleosome-free regions
- Cancer-specific epigenetic changes may alter patterns
- Signal should show oscillating pattern (nucleosome periodicity)

---

## 4. DATA QUALITY ASSESSMENT

### 4.1 Completeness ✅

| Aspect | Status | Details |
|--------|--------|---------|
| Missing Values | ✅ EXCELLENT | 0 missing values across all datasets |
| Sample Consistency | ✅ EXCELLENT | All 300 samples present in each dataset |
| Feature Completeness | ✅ EXCELLENT | All features fully populated |
| Label Information | ✅ EXCELLENT | All samples clearly labeled |

### 4.2 Feature Redundancy ⚠️

**High Correlation Analysis:**

| Dataset | High Correlations (>0.95) | Severity |
|---------|---------------------------|----------|
| EM | 161 pairs | ⚠️ Moderate |
| FLEN | 1,463 pairs | ⚠️ High |
| NUCLEOSOME | 5,403 pairs | 🔴 Critical |

**Implications:**
- Many features are highly redundant
- Feature selection is **critical**
- PCA/LDA would be very effective
- Risk of multicollinearity in linear models

**Recommendations:**
1. Remove features with correlation > 0.95
2. Use PCA to reduce dimensionality
3. Use regularization (L1/L2) in models
4. Feature selection before training

### 4.3 Feature Variance ⚠️

**Low Variance Analysis:**

| Dataset | Low Variance Features (<1e-6) | Percentage |
|---------|-------------------------------|------------|
| EM | 253 / 256 | 98.8% |
| FLEN | 227 / 301 | 75.4% |
| NUCLEOSOME | 601 / 601 | 100% |

**Interpretation:**
- Most features have very small variance
- This is expected for frequency/proportion data
- Does NOT mean features are uninformative
- Variance is low because values are small (0-0.04 range)
- Differences between classes may still be significant

**Action Items:**
- ❌ Don't remove based on absolute variance alone
- ✅ Use statistical tests (ANOVA, t-test) for feature selection
- ✅ Check relative variance and separation between classes
- ✅ Scaling will normalize variance differences

### 4.4 Value Ranges 📊

| Dataset | Min | Max | Range | Scaling Needed |
|---------|-----|-----|-------|----------------|
| EM | 0.000054 | 0.027595 | 0.0275 | ⚠️ Yes |
| FLEN | 0.000000 | 0.041626 | 0.0416 | ⚠️ Yes |
| NUCLEOSOME | 0.000063 | 0.006160 | 0.0061 | ⚠️ Yes |

**Observations:**
- All features are positive (proportions/frequencies)
- Different scale ranges across datasets
- NUCLEOSOME has much smaller values
- Need standardization for fair comparison

**Recommendations:**
- Use `StandardScaler` to normalize to mean=0, std=1
- Or use `MinMaxScaler` to scale to [0,1]
- Apply scaling **after** train/test split
- Fit scaler on training data only

---

## 5. FEATURE SPACE CHARACTERISTICS

### 5.1 Dimensionality

**Feature-to-Sample Ratio:**
```
Total Features: 1,158
Total Samples:  300
Ratio: 3.86:1
```

**Curse of Dimensionality:**
- 🔴 **High risk** - Features > Samples
- Classic "large p, small n" problem
- Models can easily overfit
- Feature selection is essential

**Mitigation Strategies:**
1. **Aggressive feature selection**
   - Keep only top 50-100 most important features
   - Use statistical tests (ANOVA F-score)
   - Use tree-based feature importance

2. **Dimensionality reduction**
   - PCA to reduce to 50-100 components (preserving 95-99% variance)
   - LDA for supervised reduction (max 5 components for 6 classes)

3. **Regularization**
   - L1 (Lasso) for sparse solutions
   - L2 (Ridge) for coefficient shrinkage
   - Elastic Net for combination

4. **Ensemble methods**
   - Random Forest naturally handles high dimensions
   - XGBoost with feature subsampling

### 5.2 Feature Composition

**Combined Feature Set:**
```
EM Features:         256 (22.1%)
FLEN Features:       301 (26.0%)
NUCLEOSOME Features: 601 (51.9%)
────────────────────────────────
TOTAL:             1,158 (100%)
```

**Feature Type Distribution:**
- Nucleosome features dominate (>50%)
- All three types are comparable in magnitude
- May need to evaluate each feature set independently

**Recommendations:**
1. **Train models on each feature set separately**
   - EM-only model
   - FLEN-only model
   - NUCLEOSOME-only model
   
2. **Compare performance:**
   - Which feature set is most informative?
   - Are they complementary or redundant?
   
3. **Feature set fusion:**
   - Early fusion: Concatenate all features
   - Late fusion: Ensemble predictions
   - Weighted fusion: Weight by importance

---

## 6. PRELIMINARY INSIGHTS

### 6.1 Expected Challenges

| Challenge | Severity | Impact | Mitigation |
|-----------|----------|--------|------------|
| High dimensionality | 🔴 High | Overfitting | Feature selection, PCA, regularization |
| Class imbalance | ⚠️ Medium | Biased predictions | Stratified CV, class weights, SMOTE |
| Feature correlation | 🔴 High | Multicollinearity | Remove correlated features, PCA |
| Small sample size | ⚠️ Medium | Limited generalization | Cross-validation, regularization |
| Multiple feature types | ⚠️ Medium | Optimal fusion unclear | Test separately, then combine |

### 6.2 Strengths

✅ **Excellent data quality:**
- No missing values
- Consistent across all datasets
- All samples properly labeled

✅ **Rich feature representation:**
- Multiple complementary views of cfDNA
- Captures different biological aspects
- Proven biomarkers in literature

✅ **Balanced classes (relatively):**
- No extreme imbalance
- Control vs Disease is well-balanced
- Suitable for multi-class classification

✅ **Biological relevance:**
- Features have clear biological interpretation
- Published studies support these markers
- Clinical applicability

### 6.3 Potential Discriminative Features

Based on biological knowledge, we expect these features to be important:

**EM Features:**
- Specific motifs enriched in cancer (e.g., CCCA, CCAG)
- Motifs associated with apoptotic DNA fragmentation

**FLEN Features:**
- Peak height at ~167bp (mono-nucleosomal)
- Ratio of short/long fragments
- Overall distribution shape

**NUCLEOSOME Features:**
- Coverage around TSS (±50bp)
- Nucleosome depletion signal
- Periodicity patterns

---

## 7. DATA PREPROCESSING REQUIREMENTS

### 7.1 Essential Preprocessing Steps

```python
# 1. Data Transposition
X_em = em_df.T          # (300, 256)
X_flen = flen_df.T      # (300, 301)
X_nuc = nucleosome_df.T # (300, 601)

# 2. Label Extraction
y = extract_labels_from_columns(em_df.columns)
# 0: Control, 1: Breast, 2: CRC, 3: Gastric, 4: Liver, 5: Lung

# 3. Feature Combination
X_combined = np.concatenate([X_em, X_flen, X_nuc], axis=1)
# Shape: (300, 1158)

# 4. Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Feature Selection
selector = SelectKBest(f_classif, k=100)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
```

### 7.2 Recommended Pipeline

```
Raw Data
    ↓
[Transpose] - Convert to samples×features
    ↓
[Label Encoding] - Create target variable
    ↓
[Train/Test Split] - Stratified 80/20
    ↓
[Feature Scaling] - StandardScaler (fit on train only)
    ↓
[Feature Selection] - Statistical or model-based
    ↓
[Model Training] - With cross-validation
    ↓
[Evaluation] - On held-out test set
```

### 7.3 Alternative Approaches

**Approach A: Feature Selection First**
```
1. Combine all features
2. Select top 100 features (ANOVA)
3. Train models on selected features
```
**Pros:** Simpler, faster  
**Cons:** May lose important interactions

**Approach B: Dimensionality Reduction First**
```
1. Combine all features
2. PCA to 100 components
3. Train models on PC scores
```
**Pros:** Handles correlation, reduces noise  
**Cons:** Loses interpretability

**Approach C: Feature Set Modeling**
```
1. Train separate models for EM, FLEN, NUCLEOSOME
2. Ensemble the predictions
```
**Pros:** Leverages each feature type's strength  
**Cons:** More complex, computationally expensive

---

## 8. RECOMMENDED ANALYSIS WORKFLOW

### Phase 1: Exploratory Visualization (1-2 hours)

1. **PCA Visualization**
   - 2D scatter plot of first 2 PCs
   - Color by class
   - Assess class separability

2. **t-SNE Visualization**
   - Non-linear dimensionality reduction
   - Better for cluster visualization
   - Perplexity = 30-50

3. **Feature Distributions**
   - Box plots for top features by ANOVA
   - Violin plots by class
   - Identify discriminative patterns

4. **Correlation Heatmaps**
   - Within each feature set
   - Identify redundant features
   - Guide feature selection

### Phase 2: Feature Engineering & Selection (2-3 hours)

1. **Statistical Feature Selection**
   - ANOVA F-test for each feature
   - Select top 50-100 by p-value
   - Mutual information score

2. **Correlation-based Filtering**
   - Remove features with r > 0.95
   - Keep one from each correlated group

3. **PCA Analysis**
   - Determine number of components for 95%, 99% variance
   - Analyze component loadings
   - Identify key contributing features

4. **Feature Importance from Trees**
   - Train Random Forest on all features
   - Extract feature_importances_
   - Select top features

### Phase 3: Baseline Modeling (2-3 hours)

Test simple models with default parameters:

1. Logistic Regression (L2)
2. Random Forest (100 trees)
3. SVM (RBF kernel)
4. KNN (k=5)

**Metrics to track:**
- Accuracy
- Macro F1-score
- Per-class F1-scores
- Confusion matrix
- 5-fold CV scores

### Phase 4: Strategy Implementation (4-6 hours)

**Strategy 1: Two-Stage**
- Stage 1: Binary (Control vs Disease)
- Stage 2: 5-class (Cancer types)
- Evaluate cascade error

**Strategy 2: Direct 6-Class**
- Single multi-class model
- Compare multiple algorithms
- Hyperparameter tuning

### Phase 5: Model Optimization (3-4 hours)

1. **Hyperparameter Tuning**
   - Grid Search or Random Search
   - Nested cross-validation
   - Focus on top 2-3 models

2. **Ensemble Methods**
   - Voting classifier
   - Stacking
   - Weighted averaging

3. **Final Evaluation**
   - Hold-out test set
   - Bootstrap confidence intervals
   - Statistical comparison of models

---

## 9. SUCCESS METRICS

### 9.1 Model Performance Targets

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Overall Accuracy | 70% | 80% | 90% |
| Macro F1-Score | 0.65 | 0.75 | 0.85 |
| Control Detection (Sensitivity) | 80% | 90% | 95% |
| Cancer Detection (Specificity) | 80% | 90% | 95% |
| Per-class F1 (minimum) | 0.60 | 0.70 | 0.80 |

### 9.2 Clinical Relevance

**Most Important:**
1. **High Sensitivity for Control** - Don't miss healthy people
2. **High Specificity for Cancer** - Don't false alarm
3. **Balanced per-class performance** - Don't favor one cancer type

**Secondary:**
4. Model interpretability
5. Computational efficiency
6. Robustness across CV folds

---

## 10. POTENTIAL RISKS & LIMITATIONS

### 10.1 Data Limitations

⚠️ **Small sample size (n=300)**
- Limited statistical power
- Risk of overfitting
- May not generalize well
- Confidence intervals will be wide

⚠️ **Class imbalance (Liver only 30 samples)**
- May have poor Liver classification
- Need careful validation
- Consider collecting more Liver samples

⚠️ **High dimensionality (p=1,158)**
- Curse of dimensionality
- Need aggressive feature selection
- Many models may overfit

### 10.2 Modeling Risks

⚠️ **Strategy 1 (Two-stage)**
- Cascade error propagation
- Stage 1 error affects Stage 2
- Overall performance is product of both stages

⚠️ **Strategy 2 (6-class)**
- Harder optimization problem
- May need more data
- Confusion between similar cancer types

⚠️ **Generalization**
- No external validation set
- Performance on new data unknown
- Batch effects between studies

### 10.3 Biological Considerations

⚠️ **Tumor heterogeneity**
- Different subtypes within cancer types
- Stage variations not captured
- Grade variations not accounted for

⚠️ **Confounding factors**
- Age, sex, ethnicity not in data
- Comorbidities unknown
- Treatment status unclear

---

## 11. NEXT STEPS

### Immediate Actions

1. ✅ **Create data preprocessing pipeline**
   - Transpose datasets
   - Extract labels
   - Combine features
   - Split train/test

2. ✅ **Generate visualizations**
   - PCA plots
   - t-SNE plots
   - Feature distributions
   - Correlation heatmaps

3. ✅ **Baseline modeling**
   - Train 4-5 simple models
   - 5-fold cross-validation
   - Compare performances

### Short-term (This Week)

4. **Feature selection experiments**
   - Statistical selection
   - PCA reduction
   - Tree-based importance
   - Compare methods

5. **Strategy 1 implementation**
   - Train Stage 1 model
   - Train Stage 2 model
   - Evaluate cascade

6. **Strategy 2 implementation**
   - Train 6-class models
   - Hyperparameter tuning
   - Compare with Strategy 1

### Medium-term (Next Steps)

7. **Model optimization**
   - Extensive hyperparameter search
   - Ensemble methods
   - Final model selection

8. **Documentation**
   - Write comprehensive report
   - Create visualizations
   - Document all findings

9. **Deliverables**
   - Jupyter notebook
   - Final models (saved)
   - Performance report
   - Recommendations

---

## 12. SUMMARY & RECOMMENDATIONS

### Key Findings

✅ **Excellent data quality** - No missing values, consistent structure  
⚠️ **Moderate class imbalance** - Liver underrepresented  
🔴 **High dimensionality** - 1,158 features for 300 samples  
⚠️ **Feature redundancy** - High correlations, need selection  
✅ **Multiple feature views** - Complementary biological information  

### Critical Success Factors

1. **Feature selection** is ESSENTIAL - Cannot use all 1,158 features
2. **Stratified sampling** is MANDATORY - Preserve class balance
3. **Cross-validation** is CRITICAL - Avoid overfitting
4. **Regularization** is IMPORTANT - Control model complexity
5. **Proper evaluation** is KEY - Use appropriate metrics

### Recommended Strategy

**Primary Approach:**
- Start with **Strategy 2 (6-class direct classification)**
- Simpler conceptually
- No cascade error
- Easier to optimize

**If Strategy 2 fails:**
- Fall back to **Strategy 1 (Two-stage)**
- May achieve better binary discrimination
- Can optimize each stage independently

**Feature Selection:**
- Use **ANOVA F-test** to select top 100 features
- Validate with **PCA** (keep 95% variance)
- Compare both approaches

**Model Selection:**
- Primary: **Random Forest** and **XGBoost**
- Secondary: **SVM** and **Logistic Regression**
- Ensemble: Combine top 2-3 models

### Expected Outcomes

**Realistic Performance:**
- Overall accuracy: 75-85%
- Macro F1-score: 0.70-0.80
- Control detection: 85-95%
- Cancer type discrimination: 70-80%

**Best Case:**
- Accuracy > 90%
- All classes F1 > 0.85
- Perfect Control vs Disease separation

**Worst Case:**
- Accuracy < 70%
- Poor Liver classification
- Confusion between cancer types
- Need more data or better features

---

## APPENDIX

### A. Feature Name Examples

**EM Features (256 total):**
```
CCCA, CCAG, CCTG, CAAA, CCAA, CCCT, CCAT, CCTT,
TGAA, CCTC, CCGA, ACAA, TGCA, TGTG, TGAG, ...
```

**FLEN Features (301 total):**
```
50, 51, 52, 53, 54, 55, ..., 345, 346, 347, 348, 349, 350
```

**NUCLEOSOME Features (601 total):**
```
-300, -299, -298, ..., -2, -1, 0, 1, 2, ..., 298, 299, 300
```

### B. Statistical Summary Tables

**EM Feature Statistics:**
| Statistic | Control_1 | Breast_1 | CRC_1 | Gastric_1 | Liver_1 | Lung_1 |
|-----------|-----------|----------|-------|-----------|---------|--------|
| Mean | 0.00391 | 0.00391 | 0.00391 | 0.00391 | 0.00391 | 0.00391 |
| Std | 0.00342 | 0.00353 | 0.00349 | 0.00351 | 0.00345 | 0.00348 |
| Min | 0.00010 | 0.00008 | 0.00009 | 0.00009 | 0.00008 | 0.00009 |
| Max | 0.02327 | 0.02479 | 0.02496 | 0.02470 | 0.02364 | 0.02475 |

### C. Data File Information

**File Sizes:**
- EM.csv: 1.4 MB
- FLEN.csv: 1.2 MB  
- NUCLEOSOME.csv: 3.3 MB
- **Total: 5.9 MB**

**Compression Potential:**
- Current: Plain text CSV
- Compressed: ~500 KB (estimated)
- Binary format: ~300 KB (estimated)

### D. Biological Background

**Cell-Free DNA (cfDNA):**
- DNA fragments circulating in bloodstream
- Released from dying cells (apoptosis, necrosis)
- Tumor cells release cfDNA with unique characteristics
- Can be used as liquid biopsy for cancer detection

**End Motifs:**
- Specific DNA sequences at fragment ends
- Reflect enzymatic DNA cleavage preferences
- Different in cancer vs normal cells

**Fragment Length:**
- Tumor DNA often shorter than normal DNA
- Nucleosome-sized fragments (~167 bp)
- Can distinguish cancer from healthy

**Nucleosome Positioning:**
- DNA packaging affects fragmentation
- Cancer alters chromatin structure
- TSS regions show characteristic patterns

---

**End of Report**

Generated: January 31, 2026  
Version: 1.0  
Status: Ready for Analysis
