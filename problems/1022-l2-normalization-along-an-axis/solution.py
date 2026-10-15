import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> list:
    """
    L2-normalize x along the given axis.

    Args:
        x: input NumPy array
        axis: axis along which to normalize
        eps: small constant for numerical stability

    Returns:
        Normalized earray as a nested Python list.
    """
    sume= np.sum(x ** 2, axis=axis, keepdims=True)
    de = np.sqrt(sume + eps)
    return x / de
    
