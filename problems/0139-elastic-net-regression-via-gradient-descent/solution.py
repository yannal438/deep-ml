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
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n_samples, n_features = X.shape
    
    # Initialisation des paramètres
    weights = np.zeros(n_features)
    bias = 0.0
    cost_history = []
    
    for i in range(max_iter):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        
        # Fonction de coût avec la formulation 1/(2*N) pour la MSE
        MSE = (1.0 / (2.0 * n_samples)) * np.sum(error**2)
        L1 = alpha1 * np.sum(np.abs(weights))
        L2 = alpha2 * np.sum(weights**2)
        cost = MSE + L1 + L2
        cost_history.append(cost)
        
        # Gradients associés (Division par n_samples au lieu de 2/n_samples)
        gradient_w = (1.0 / n_samples) * (X.T @ error) + alpha1 * np.sign(weights) + 2.0 * alpha2 * weights
        gradient_b = (1.0 / n_samples) * np.sum(error)
        
        # Mises à jour des poids et du biais
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b
        
        # Critère d'arrêt
        if np.linalg.norm(gradient_w) < tol:
            break
            
    return weights, bias

