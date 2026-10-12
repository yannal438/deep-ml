import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)
    # squared bias 
    prediction = np.mean(predictions, axis= 0)
    squared_bias= np.mean((prediction - y_true)**2)
    # Variance 
    ecart= (predictions - prediction)**2
    variance = np.mean(ecart)
    #MSE 
    MSE = np.mean((predictions - y_true)**2)
    return {
        'bias_squared':float(squared_bias.item()),
        'variance': float(variance.item()),
        'mse': float(MSE.item())
    }

