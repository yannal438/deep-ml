import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code her
    S = np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
    return S

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    for i in range(max_iter):
        weights_old = weights.copy()
        y_pred = np.dot(X, weights) + bias
        erreur = y_pred - y 
        gradient_w = (1.0 / n_samples) * (X.T @ erreur)
        gradient_b = (1.0 / n_samples) * np.sum(erreur)
        weights_temps = weights - learning_rate * gradient_w 
        weights = soft_threshold(weights_temps, learning_rate * alpha)
        bias = bias - learning_rate * gradient_b 
        w_change = np.max(np.abs(weights) - weights_old)
        if w_change <= tol:
            break 
    return  weights, float(bias)
    
    # Your code here
    pass
