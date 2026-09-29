import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[int|float]:

    a_np = np.array(a)
    return a_np.T.tolist()
