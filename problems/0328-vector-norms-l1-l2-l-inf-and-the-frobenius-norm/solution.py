import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    type = {'l1', 'l2', 'linf', 'frobenius'}
    arr_n = arr.ravel()
    if norm_type == 'l1':
        return float(np.linalg.norm(arr_n, ord= 1))
    elif norm_type == 'l2':
        return float(np.linalg.norm(arr_n, ord=2))
    elif norm_type == 'linf':
        return  float(np.max(np.abs(arr_n)))
    elif norm_type == 'frobenius':
        if arr.ndim != 2:
            raise ValueError("Frobenius norm is a matrix norm and requires a 2D array.")
        return float(np.sqrt(np.sum(arr_n ** 2)))
    else:
        return 'ValueError'