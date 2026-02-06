#!/usr/bin/env python3
"""
Q2. Basic Statistics - P-value and Multiple Testing Corrections

This script demonstrates:
1. What a p-value means
2. The multiple testing problem
3. Correction methods: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg

Author: Solution for Q2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simulate_pvalue_meaning():
    """
    Demonstrate what p-value means through simulation.
    """
    print("\n" + "="*70)
    print(" PART 1: WHAT IS A P-VALUE? ")
    print("="*70)
    
    print("""
DEFINITION:
A p-value is the probability of observing data as extreme as (or more extreme than)
what was observed, assuming the null hypothesis H₀ is true.

  p = P(data as extreme as observed | H₀)

INTERPRETATION:
  - Small p (≤ 0.05): Data unlikely under H₀ → evidence against H₀
  - Large p (> 0.05): Data plausible under H₀ → insufficient evidence to reject

IMPORTANT:
  A p-value is NOT the probability that H₀ is true!
  It measures how surprising the data would be IF H₀ were true.
""")

    # Example: t-test simulation
    np.random.seed(42)
    
    # Null hypothesis is true (both groups from same distribution)
    group1 = np.random.normal(loc=0, scale=1, size=30)
    group2 = np.random.normal(loc=0, scale=1, size=30)
    _, p_null = stats.ttest_ind(group1, group2)
    
    # Alternative is true (different means)
    group3 = np.random.normal(loc=0, scale=1, size=30)
    group4 = np.random.normal(loc=1.5, scale=1, size=30)  # Different mean
    _, p_alt = stats.ttest_ind(group3, group4)
    
    print("EXAMPLE: Two-sample t-test")
    print("-" * 40)
    print(f"Case 1 (H₀ true, same populations):  p = {p_null:.4f}")
    print(f"Case 2 (H₁ true, different means):   p = {p_alt:.6f}")


def simulate_multiple_testing_problem():
    """
    Demonstrate the multiple testing problem through simulation.
    """
    print("\n" + "="*70)
    print(" PART 2: THE MULTIPLE TESTING PROBLEM ")
    print("="*70)
    
    print("""
PROBLEM:
If we run one test at α = 0.05, we accept 5% false positive rate.
If we run m independent tests, the probability of ≥1 false positive is:

  P(≥1 FP) = 1 - (1 - α)^m

For m = 100 tests at α = 0.05:
  P(≥1 FP) = 1 - (0.95)^100 ≈ 0.994

Almost guaranteed to get a false positive!
""")

    # Calculate theoretical probabilities
    alpha = 0.05
    m_values = [1, 5, 10, 20, 50, 100, 1000]
    
    print("Number of tests (m) | P(≥1 False Positive)")
    print("-" * 45)
    for m in m_values:
        prob_fp = 1 - (1 - alpha) ** m
        print(f"{m:>18} | {prob_fp:.4f}")
    
    # Simulate to verify
    print("\n" + "-"*45)
    print("SIMULATION VERIFICATION (10,000 experiments):")
    print("-"*45)
    
    np.random.seed(42)
    n_experiments = 10000
    m = 100  # 100 tests per experiment
    
    false_positive_count = 0
    for _ in range(n_experiments):
        # Generate 100 p-values under null hypothesis (uniform distribution)
        pvalues = np.random.uniform(0, 1, m)
        # Check if any p-value < 0.05
        if np.any(pvalues < 0.05):
            false_positive_count += 1
    
    simulated_prob = false_positive_count / n_experiments
    theoretical_prob = 1 - (1 - 0.05) ** m
    
    print(f"Simulated P(≥1 FP | m=100):   {simulated_prob:.4f}")
    print(f"Theoretical P(≥1 FP | m=100): {theoretical_prob:.4f}")


def demonstrate_correction_methods():
    """
    Demonstrate Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg corrections.
    """
    print("\n" + "="*70)
    print(" PART 3: MULTIPLE TESTING CORRECTION METHODS ")
    print("="*70)
    
    # Example p-values (some significant, some not)
    np.random.seed(42)
    m = 20
    
    # Mix of true nulls and true alternatives
    pvalues_null = np.random.uniform(0.1, 1.0, 15)  # 15 true nulls
    pvalues_alt = np.array([0.001, 0.005, 0.01, 0.02, 0.03])  # 5 true alternatives
    pvalues = np.concatenate([pvalues_null, pvalues_alt])
    np.random.shuffle(pvalues)
    
    alpha = 0.05
    
    print(f"\nExample: {m} hypothesis tests at α = {alpha}")
    print(f"P-values: {np.sort(pvalues)[:5].round(4)}... (showing first 5 sorted)")
    
    # Method 1: Bonferroni
    print("\n" + "-"*60)
    print("METHOD 1: BONFERRONI (controls FWER)")
    print("-"*60)
    print(f"  Adjusted threshold: α/m = {alpha}/{m} = {alpha/m:.4f}")
    bonf_reject = pvalues < (alpha / m)
    print(f"  Rejected: {np.sum(bonf_reject)} hypotheses")
    print("  ✓ Very conservative - controls family-wise error rate (FWER)")
    
    # Method 2: Holm-Bonferroni
    print("\n" + "-"*60)
    print("METHOD 2: HOLM-BONFERRONI (controls FWER, less conservative)")
    print("-"*60)
    print("  Step-down procedure:")
    print("  1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)")
    print("  2. For k-th smallest p-value, compare to α/(m-k+1)")
    print("  3. Reject until p(k) > threshold")
    
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    holm_reject = np.zeros(m, dtype=bool)
    
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_pvalues[k] <= threshold:
            holm_reject[sorted_indices[k]] = True
        else:
            break
    
    print(f"  Rejected: {np.sum(holm_reject)} hypotheses")
    
    # Method 3: Benjamini-Hochberg
    print("\n" + "-"*60)
    print("METHOD 3: BENJAMINI-HOCHBERG (controls FDR)")
    print("-"*60)
    print("  Step-up procedure:")
    print("  1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)")
    print("  2. Find largest k where p(k) ≤ (k/m) × q")
    print("  3. Reject p(1), ..., p(k)")
    
    q = 0.05  # FDR level
    sorted_pvalues = np.sort(pvalues)
    bh_reject_count = 0
    
    for k in range(m, 0, -1):
        threshold = (k / m) * q
        if sorted_pvalues[k-1] <= threshold:
            bh_reject_count = k
            break
    
    print(f"  Rejected: {bh_reject_count} hypotheses")
    print("  ✓ Less conservative - controls false discovery rate (FDR)")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Method':<25} {'Rejections':<15} {'Controls':<15}")
    print("-"*60)
    print(f"{'No correction':<25} {np.sum(pvalues < 0.05):<15} {'Nothing!':<15}")
    print(f"{'Bonferroni':<25} {np.sum(bonf_reject):<15} {'FWER':<15}")
    print(f"{'Holm-Bonferroni':<25} {np.sum(holm_reject):<15} {'FWER':<15}")
    print(f"{'Benjamini-Hochberg':<25} {bh_reject_count:<15} {'FDR':<15}")


def plot_multiple_testing_visualization():
    """Create visualization of multiple testing correction comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Family-wise error rate vs number of tests
    alpha = 0.05
    m_range = np.arange(1, 201)
    fwer = 1 - (1 - alpha) ** m_range
    
    axes[0].plot(m_range, fwer, 'b-', linewidth=2)
    axes[0].axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
    axes[0].axhline(y=0.994, color='g', linestyle='--', label='P(≥1 FP) at m=100')
    axes[0].set_xlabel('Number of tests (m)')
    axes[0].set_ylabel('P(at least 1 False Positive)')
    axes[0].set_title('Multiple Testing: Family-Wise Error Rate')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Right plot: Comparison of correction thresholds
    m = 20
    k_values = np.arange(1, m + 1)
    
    # Uncorrected threshold
    uncorrected = np.full(m, 0.05)
    
    # Bonferroni threshold
    bonferroni = np.full(m, 0.05 / m)
    
    # Holm-Bonferroni threshold (depends on rank)
    holm = 0.05 / (m - k_values + 1)
    
    # Benjamini-Hochberg threshold (depends on rank)
    bh = (k_values / m) * 0.05
    
    axes[1].step(k_values, uncorrected, 'r-', linewidth=2, where='mid', label='No correction')
    axes[1].step(k_values, bonferroni, 'b-', linewidth=2, where='mid', label='Bonferroni')
    axes[1].step(k_values, holm, 'g-', linewidth=2, where='mid', label='Holm-Bonferroni')
    axes[1].step(k_values, bh, 'm-', linewidth=2, where='mid', label='Benjamini-Hochberg')
    
    axes[1].set_xlabel('Rank of sorted p-value (k)')
    axes[1].set_ylabel('Significance threshold')
    axes[1].set_title(f'Correction Thresholds (m={m} tests, α=0.05)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('multiple_testing_corrections.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nFigure saved to: multiple_testing_corrections.png")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Q2: P-VALUES AND MULTIPLE TESTING CORRECTIONS ")
    print("="*70)
    
    # Part 1: P-value meaning
    simulate_pvalue_meaning()
    
    # Part 2: Multiple testing problem
    simulate_multiple_testing_problem()
    
    # Part 3: Correction methods
    demonstrate_correction_methods()
    
    # Part 4: Visualization
    print("\n" + "="*70)
    print(" GENERATING VISUALIZATION ")
    print("="*70)
    plot_multiple_testing_visualization()
    
    print("\n" + "="*70)
    print(" SUMMARY ")
    print("="*70)
    print("""
KEY TAKEAWAYS:

1. P-VALUE:
   - Probability of data as extreme as observed, given H₀ is true
   - NOT the probability that H₀ is true

2. MULTIPLE TESTING PROBLEM:
   - Running many tests inflates false positive rate
   - With 100 tests at α=0.05, ~99.4% chance of ≥1 false positive

3. CORRECTION METHODS:
   - Bonferroni: Most conservative, use α/m threshold
   - Holm-Bonferroni: Less conservative, step-down procedure
   - Benjamini-Hochberg: Controls FDR, recommended for discovery research

4. WHEN TO USE WHICH:
   - FWER control (Bonferroni/Holm): When ANY false positive is costly
   - FDR control (BH): When some false positives are acceptable
""")
