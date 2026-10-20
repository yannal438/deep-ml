import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    X = np.asarray(X, dtype = np.float64)
    n_samples, n_features = X.shape

    # Centrons les donnée
    X_centered = X - np.mean(X, axis=0)
    # Calculons la covariance
    covariance = np.cov(X_centered, rowvar=False)
    if covariance.ndim ==0:
        return np.array([0.1])
    # Extayons les valeurs uniques
    eigenvalues,_=np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1] # Ordre decroissant
    eigenvalues = np.maximum(eigenvalues, 0) # Eviter les valeurs négatives
    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        return np.zeros_like(eigenvalues)

    return (eigenvalues / total_variance)
