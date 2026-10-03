import numpy as np
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function4
    n_features = 30
    n_samples, n_features = X_train.shape
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype = np.float64)

    X_val = np.asarray(X_val, dtype=np.float64).flatten()
    y_val = np.asarray(y_val, dtype=np.float64).flatten()

    W = np.random.randn(n_features) * 0.01
    b = 0.0
    lr = 0.01
    iterations = 500

    for i in range(iterations):
        z = X_train @ W + b 
        prediction = 1 / (1 + np.exp(-z))
        loss = prediction - y_train
        gradient_w = (1/iterations) * (X_train.T @ loss)
        gradient_b = (1/iterations) * np.sum(loss)

        W -= lr * gradient_w
        b -= lr * gradient_b


    def predict(X):
        z_new = X @ W + b 
        probabilit_new = 1 /(1 + np.exp(-z_new))
        classe = np.where(probabilit_new>=0.5, 1, 0)
        return classe
    return predict

    
    pass
