#!/usr/bin/env python3
"""
Q3. Foundation of Machine Learning - Logistic Regression from Scratch

This script implements:
1. Logistic Regression class without using sklearn
2. Gradient descent optimization
3. Cross-entropy loss function
4. Evaluation metrics (accuracy, precision, recall, F1)

Author: Solution for Q3
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    
    Model:
        p(y=1|x) = σ(z) = 1 / (1 + exp(-z)), where z = w^T x + b
    
    Loss Function (Cross-Entropy):
        J(w,b) = -Σ [y_i log(σ(z_i)) + (1-y_i) log(1-σ(z_i))]
    
    Gradients:
        ∂J/∂w = Σ (σ(z_i) - y_i) x_i
        ∂J/∂b = Σ (σ(z_i) - y_i)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of gradient descent iterations
        verbose : bool
            If True, print cost every 100 iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        Compute the sigmoid function: σ(z) = 1 / (1 + exp(-z))
        
        Numerically stable implementation using clipping.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute the cross-entropy loss.
        
        J = -1/m * Σ [y log(ŷ) + (1-y) log(1-ŷ)]
        """
        m = len(y_true)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -1/m * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features
        y : np.ndarray, shape (n_samples,)
            Training labels (0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute and store cost
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            # ∂J/∂w = (1/n) * X^T (σ(z) - y)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            # ∂J/∂b = (1/n) * Σ(σ(z) - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose and (i % 100 == 0):
                print(f"Iteration {i}: Cost = {cost:.4f}")
    
    def predict_proba(self, X):
        """Return probability of class 1 for each sample."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def accuracy_score(y_true, y_pred):
    """Compute accuracy: proportion of correct predictions."""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Returns:
        [[TN, FP],
         [FN, TP]]
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])


def classification_report(y_true, y_pred):
    """Print detailed classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              0      1")
    print(f"Actual  0   {TN:3d}   {FP:3d}")
    print(f"        1   {FN:3d}   {TP:3d}")
    print("="*50)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score}


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into train and test sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    """Standardize features: (x - mean) / std"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled


def generate_binary_classification_data(n_samples=200, n_features=2, random_state=42):
    """Generate synthetic binary classification dataset."""
    np.random.seed(random_state)
    
    n_each = n_samples // 2
    
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n_each, n_features) + np.array([-1, -1])
    y0 = np.zeros(n_each)
    
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_each, n_features) + np.array([1, 1])
    y1 = np.ones(n_each)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def load_iris_binary():
    """
    Load a binary version of Iris dataset (Setosa vs Versicolor).
    Uses only sepal length and sepal width for visualization.
    """
    # Iris dataset (simplified - first 100 samples are Setosa and Versicolor)
    # Sepal length, sepal width (first 50 = Setosa, next 50 = Versicolor)
    np.random.seed(42)
    
    # Setosa: mean=[5.0, 3.4]
    X_setosa = np.random.randn(50, 2) * [0.35, 0.38] + [5.0, 3.4]
    
    # Versicolor: mean=[5.9, 2.8]
    X_versicolor = np.random.randn(50, 2) * [0.52, 0.31] + [5.9, 2.8]
    
    X = np.vstack([X_setosa, X_versicolor])
    y = np.array([0]*50 + [1]*50)
    
    # Shuffle
    idx = np.random.permutation(100)
    return X[idx], y[idx]


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot the decision boundary for 2D data."""
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu_r')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Scatter plot of data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu_r', 
                          edgecolors='black', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter, label='Probability of Class 1')
    plt.tight_layout()


def plot_cost_history(cost_history):
    """Plot the cost function over training iterations."""
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss over Iterations')
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_bias_variance_tradeoff():
    """Create the bias-variance tradeoff plot for Q3.1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0.1, 10, 200)
    
    # Define curves
    bias_squared = 6 / (x + 1) + 0.2
    variance = 0.05 * (x ** 2) + 0.2
    noise = np.full_like(x, 0.5)
    test_error = bias_squared + variance + 0.15
    
    ax.plot(x, bias_squared, 'b-', linewidth=2, label='Bias² (decreases)')
    ax.plot(x, variance, 'r-', linewidth=2, label='Variance (increases)')
    ax.plot(x, noise, 'k--', linewidth=1.5, label='Irreducible Noise')
    ax.plot(x, test_error, 'g-', linewidth=2.5, label='Test Error (U-shape)')
    
    # Mark optimal complexity
    optimal_idx = np.argmin(test_error)
    ax.axvline(x=x[optimal_idx], color='gray', linestyle=':', alpha=0.7)
    ax.annotate('Optimal\nComplexity', xy=(x[optimal_idx], 0.5), 
                xytext=(x[optimal_idx] + 1, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)
    
    # Labels for regions
    ax.text(1.5, 5, 'UNDERFITTING\n(High Bias)', fontsize=10, ha='center')
    ax.text(8, 4, 'OVERFITTING\n(High Variance)', fontsize=10, ha='center')
    
    ax.set_xlabel('Model Complexity', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14)
    ax.legend(loc='upper center')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 7])
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bias_variance_plot.png', dpi=300, bbox_inches='tight')
    print("\nSaved: bias_variance_plot.png")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" LOGISTIC REGRESSION FROM SCRATCH - TUTORIAL ")
    print("="*70)
    
    # ========== Q3.1: Bias-Variance Tradeoff Plot ==========
    print("\n" + "="*70)
    print("Q3.1: UNDERFITTING & OVERFITTING - BIAS-VARIANCE TRADEOFF")
    print("="*70)
    print("""
UNDERFITTING: Model too simple → High bias, poor on both train & test
OVERFITTING: Model too complex → Low train error, high test error (high variance)

The total test error decomposes as:
    Test Error = Bias² + Variance + Irreducible Noise
""")
    plot_bias_variance_tradeoff()
    
    # ========== Q3.2: Logistic Regression ==========
    print("\n" + "="*70)
    print("Q3.2: LOGISTIC REGRESSION FROM SCRATCH")
    print("="*70)
    
    print("""
WHY "REGRESSION" FOR CLASSIFICATION?
- Logistic regression performs regression on PROBABILITY (continuous)
- Model: log(p/(1-p)) = w^T x + b  (log-odds are linear)
- Classification is obtained by thresholding: ŷ = I(p ≥ 0.5)

KEY FORMULAS:
- Sigmoid: σ(z) = 1 / (1 + exp(-z))
- Cross-Entropy Loss: J = -Σ[y log(ŷ) + (1-y) log(1-ŷ)]
- Gradient: ∂J/∂w = Σ(σ(z) - y) × x  ("prediction - label" × input)
""")
    
    # ========== Example 1: Synthetic Data ==========
    print("\n" + "="*70)
    print("EXAMPLE 1: SYNTHETIC BINARY CLASSIFICATION")
    print("="*70)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200, random_state=42)
    print(f"\nDataset size: {len(y)} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"Features: {X.shape[1]}")
    
    # Standardize
    X_train_scaled, X_test_scaled = standardize(X_train, X_test)
    
    # Train
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(learning_rate=0.5, n_iterations=1000, verbose=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training set
    print("\n--- Training Set Performance ---")
    y_train_pred = model.predict(X_train_scaled)
    classification_report(y_train, y_train_pred)
    
    # Evaluate on test set
    print("\n--- Test Set Performance ---")
    y_test_pred = model.predict(X_test_scaled)
    classification_report(y_test, y_test_pred)
    
    # ========== Example 2: Iris-like Data ==========
    print("\n" + "="*70)
    print("EXAMPLE 2: IRIS DATASET (BINARY CLASSIFICATION)")
    print("="*70)
    
    # Load data
    X_iris, y_iris = load_iris_binary()
    print(f"\nDataset: Iris (Setosa vs Versicolor)")
    print(f"Features: Sepal Length, Sepal Width (standardized)")
    
    # Split
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(y_train_iris)} samples")
    print(f"Test set: {len(y_test_iris)} samples")
    
    # Standardize
    X_train_iris_scaled, X_test_iris_scaled = standardize(X_train_iris, X_test_iris)
    
    # Train
    print("\nTraining Logistic Regression...")
    model_iris = LogisticRegression(learning_rate=0.5, n_iterations=1000, verbose=True)
    model_iris.fit(X_train_iris_scaled, y_train_iris)
    
    # Evaluate
    print("\n--- Training Set Performance ---")
    y_train_iris_pred = model_iris.predict(X_train_iris_scaled)
    classification_report(y_train_iris, y_train_iris_pred)
    
    print("\n--- Test Set Performance ---")
    y_test_iris_pred = model_iris.predict(X_test_iris_scaled)
    classification_report(y_test_iris, y_test_iris_pred)
    
    # ========== Visualizations ==========
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot decision boundary
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    # Decision boundary for synthetic data
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu_r')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, 
                cmap='RdYlBu_r', edgecolors='black', s=50)
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.title('Synthetic Data: Decision Boundary')
    
    plt.subplot(1, 2, 2)
    # Cost history
    plt.plot(model.cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss Convergence')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: logistic_regression_demo.png")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. SIGMOID maps any real number to (0, 1)
2. CROSS-ENTROPY is the negative log-likelihood of Bernoulli model
3. GRADIENT has elegant form: (prediction - label) × input
4. REGULARIZATION adds λ/2 ||w||² to prevent overfitting

Learned weights and bias:
""")
    print(f"  Model 1 (Synthetic): w = {model.weights.round(4)}, b = {model.bias:.4f}")
    print(f"  Model 2 (Iris):      w = {model_iris.weights.round(4)}, b = {model_iris.bias:.4f}")
