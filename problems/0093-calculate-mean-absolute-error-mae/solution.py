import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Your code here
    y_true_n = y_true.flatten()
    y_pred_n = y_pred.flatten()
    n = len(y_true_n)
    pres = np.abs(y_true_n - y_pred_n)
    mae = (1 /n) * np.sum(pres)
    return mae