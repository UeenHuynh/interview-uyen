#!/usr/bin/env python3
"""
Q1. Foundation of Data Science - Curse of Dimensionality Demonstration

Task: Generate 1000 points on unit sphere in 3D and 100D, create histograms
of pairwise distances, and explain the Curse of Dimensionality.

Author: Solution for Q1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


def generate_sphere_points(n_points: int, dim: int) -> np.ndarray:
    """
    Generate n_points approximately uniformly on the unit sphere S^{dim-1}
    by sampling Gaussian vectors then normalizing each vector.
    """
    points = np.random.randn(n_points, dim)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def compute_pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute all pairwise Euclidean distances."""
    return pdist(points, metric="euclidean")


def plot_distance_histograms(distances_3d: np.ndarray, distances_100d: np.ndarray) -> None:
    """Create side-by-side histograms for 3D and 100D distance distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3D histogram
    axes[0].hist(distances_3d, bins=50, alpha=0.7, edgecolor="black", color="#4C78A8")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distance Distribution: 3D Unit Sphere")
    axes[0].axvline(np.mean(distances_3d), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(distances_3d):.3f}")
    axes[0].axvline(np.median(distances_3d), color="green", linestyle="--", linewidth=2,
                    label=f"Median: {np.median(distances_3d):.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 100D histogram
    axes[1].hist(distances_100d, bins=50, alpha=0.7, edgecolor="black", color="#F58518")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distance Distribution: 100D Unit Sphere")
    axes[1].axvline(np.mean(distances_100d), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(distances_100d):.3f}")
    axes[1].axvline(np.median(distances_100d), color="green", linestyle="--", linewidth=2,
                    label=f"Median: {np.median(distances_100d):.3f}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("curse_of_dimensionality.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nFigure saved to: curse_of_dimensionality.png")


def compute_statistics(distances: np.ndarray, dim: int) -> None:
    """Print detailed statistics for distance distribution."""
    print(f"\n{'='*60}")
    print(f"Statistics for {dim}D Unit Sphere")
    print(f"{'='*60}")
    print(f"Number of point pairs: {len(distances):,}")
    print(f"Mean distance: {np.mean(distances):.6f}")
    print(f"Median distance: {np.median(distances):.6f}")
    print(f"Std deviation: {np.std(distances):.6f}")
    print(f"Min distance: {np.min(distances):.6f}")
    print(f"Max distance: {np.max(distances):.6f}")
    print(f"Coefficient of Variation (CV): {np.std(distances)/np.mean(distances):.6f}")
    print(f"{'='*60}")


def print_curse_of_dimensionality_explanation():
    """Print explanation of the Curse of Dimensionality."""
    print("\n" + "="*70)
    print(" THE CURSE OF DIMENSIONALITY ")
    print("="*70)
    
    print("""
OBSERVATION FROM HISTOGRAMS:
- In 3D: Distances are spread out between 0 and ~2
- In 100D: Distances sharply concentrate around sqrt(2) ≈ 1.414

MATHEMATICAL EXPLANATION:
For points x, y on unit sphere S^{d-1}:
  ||x - y||² = ||x||² + ||y||² - 2x·y = 2 - 2cos(θ)

As d → ∞:
  - E[x·y] = 0 (random vectors become orthogonal)
  - Var(x·y) = 1/d → 0 (concentration of measure)
  - ||x - y||² → 2, so ||x - y|| → sqrt(2)

EFFECTS ON MACHINE LEARNING:
1. DATA SPARSITY:
   - Fixed samples become sparse in high-D space
   - Required samples grow exponentially with dimensions

2. DISTANCE CONCENTRATION:
   - Nearest and farthest neighbors become similar distance
   - Breaks distance-based algorithms (kNN, clustering, kernels)

3. OVERFITTING RISK:
   - More features = more degrees of freedom
   - Models can fit noise, leading to poor generalization

MITIGATION STRATEGIES:
1. Dimensionality reduction (PCA, autoencoders)
2. Feature selection (filter, wrapper, embedded methods)
3. Regularization (L1, L2, elastic net)
4. Collect more data (exponentially more in high-D)
5. Use domain knowledge to select relevant features
""")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    n_points = 1000

    print("\n" + "="*70)
    print(" Q1: CURSE OF DIMENSIONALITY DEMONSTRATION ")
    print("="*70)
    print(f"\nGenerating {n_points} random points on unit spheres...")

    # Generate points
    points_3d = generate_sphere_points(n_points, dim=3)
    points_100d = generate_sphere_points(n_points, dim=100)

    # Compute pairwise distances
    distances_3d = compute_pairwise_distances(points_3d)
    distances_100d = compute_pairwise_distances(points_100d)

    # Print statistics
    compute_statistics(distances_3d, 3)
    compute_statistics(distances_100d, 100)

    # Mathematical intuition
    print("\n" + "="*60)
    print("MATHEMATICAL INTUITION:")
    print("="*60)
    print(f"Theoretical limit sqrt(2): {np.sqrt(2):.6f}")
    print(f"Our 100D mean distance:   {np.mean(distances_100d):.6f}")
    print(f"Difference from theory:   {abs(np.mean(distances_100d) - np.sqrt(2)):.6f}")

    # Plot histograms
    plot_distance_histograms(distances_3d, distances_100d)

    # Explanation
    print_curse_of_dimensionality_explanation()
