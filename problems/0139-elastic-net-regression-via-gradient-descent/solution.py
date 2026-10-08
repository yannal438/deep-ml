import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    n_samples, n_features = X.shape
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    weights = np.zeros(n_features)
    cost_history = []
    bias = 0.0
    # Fonction cout
    for i in range(max_iter):
        y_pred = np.dot(X, weights) + bias
        erreur = (y_pred - y)**2
        error = y_pred - y
        MSE = (1 / 2.0 * n_samples) * erreur 
        L1 = alpha1 * np.sum(np.abs(weights))
        L2 = alpha2 * np.sum(weights**2)
        cost = MSE + L1 + L2
        num1 = (1 / n_samples) * X.T @ (error)
        num2 = alpha1 * np.sign(weights)
        num3 = 2 * alpha2 * weights
        gradient_w = num1 +num2 + num3
        ab1 = (1 / n_samples) * np.sum(error)
        gradient_b = ab1
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b
        if np.linalg.norm(gradient_w) < tol:
            break

    return weights, bias