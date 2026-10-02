import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed=42):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    indices = np.random.default_rng(seed).permutation(n)
    train_end = int(round(n * train_frac))
    val_end = train_end + int(round(n * val_frac))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    
    baseline_value = np.median(y[train_idx])

    tes_predictions = np.full(shape=len(test_idx), fill_value=baseline_value)

    return tes_predictions, train_idx, val_idx, test_idx
    
    
    pass
