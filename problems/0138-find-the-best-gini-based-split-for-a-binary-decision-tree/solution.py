import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0
    p1 = np.mean(y)
    p0 = 1.0 - p1
    return 1.0 - (p0**2 + p1**2)

def find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """
    Finds the best feature and unique value threshold that minimizes 
    the weighted Gini impurity according to Deep ML rules.
    """
    n_samples, n_features = X.shape
    best_feature_idx = -1
    best_threshold = -1.0
    best_gini = float('inf')

    # Scan features one by one
    for feature_idx in range(n_features):
        X_column = X[:, feature_idx]
        
        # Deep ML checks unique raw values directly as thresholds
        unique_values = np.unique(X_column)
        
        for threshold in unique_values:
            left_mask = X_column <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            n_left = len(y_left)
            n_right = len(y_right)
            
            gini_left = gini(y_left)
            gini_right = gini(y_right)

            # Weighted Gini Impurity
            weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
            
            # Using strict '<' naturally preserves the very first encounter on ties
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_idx = feature_idx
                best_threshold = threshold
                
    return int(best_feature_idx), float(best_threshold)
