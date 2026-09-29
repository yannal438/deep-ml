import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    a_np = np.array(a)
    
    # Correction de la formule : on multiplie les deux dimensions du tuple (ex: 1 * 4 = 4)
    if a_np.size != new_shape[0] * new_shape[1]:
        return []
		
    reshaped = a_np.reshape(new_shape)
    return reshaped.tolist()
