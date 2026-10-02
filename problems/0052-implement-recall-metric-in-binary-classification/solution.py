import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    Returns:
        Recall value as a float
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    if y_pred.shape != y_true.shape:
        return 0.0
    
    # Correct element-wise logic:
    # True Positives (VP): where both prediction and ground truth are 1
    vp = int(np.sum((y_pred == 1) & (y_true == 1)))
    
    # False Negatives (FN): where prediction is 0 but ground truth is 1
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    if vp + fn == 0:
        return 0.0
        
    return float(vp / (vp + fn))
