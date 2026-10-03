import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    n_samples, n_features = X.shape
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    b = 0.0
    W = np.random.randn(n_features) * 0.01
    lr = 0.01
    iteration = 500

    for _ in range(iteration):
        pred = X @ W + b 
        erreur = pred - y 
        gradient_w = (1/iteration) * (X.T @ erreur)
        gradient_b = (1 / iteration) * np.sum(erreur)

        W -= lr * gradient_w
        b -= lr * gradient_b
    return W, b


    pass
