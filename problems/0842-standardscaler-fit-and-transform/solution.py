import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    moyenne = np.mean(X_train, axis=0)
    ecart_t = np.std(X_train, axis=0)
    ecart_test = np.where(ecart_t==0, 1.0, ecart_t)
    X_testt = (X_test - moyenne) / ecart_test
    return X_testt