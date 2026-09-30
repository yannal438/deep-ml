def min_max(x: list[float]) -> list[float]:
    import numpy as np
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    val_min = np.min(x, axis=0)
    val_max = np.max(x, axis=0)
    normalized = (x - val_min) / (val_max - val_min)
    return normalized

    pass