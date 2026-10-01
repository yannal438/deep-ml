import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        weights: Initial weights as a 1D array of shape (n,)
        learning_rate: Learning rate
        n_iter: Number of gradient descent iterations

    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    n_samples, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    weights = weights.reshape(-1, 1)  # Ensure weights is a column vector
    for i in range(n_iter):
        index = i % n_samples
        x_i = X[index:index +1, :]
        y_i = y[index:index + 1]

        # Compute the  prediction for the chosen
        prediction = np.dot(x_i, weights)
        error = prediction - y_i
        gradient = 2 * (x_i.T @ error)
        weights -= gradient * learning_rate
    return weights.flatten().tolist()
      
    
    