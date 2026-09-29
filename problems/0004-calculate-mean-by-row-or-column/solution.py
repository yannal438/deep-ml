import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    axes = {
        'column': 0,
        'row': 1
    }
    
    if mode not in axes:
        raise ValueError("Mode invalide. Choisissez 'column' ou 'row'.")
        
    re = axes[mode]
    means = np.mean(matrix, axis=re)
    return means.tolist()
