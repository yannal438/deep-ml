import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Convertissons les listes a et b en tableau numpy
    a_np = np.array(a)
    b_np = np.array(b)

    # Recuperation de la longueur
    length_a = a_np.shape[1]
    length_b = b_np.shape[0]

    if length_a == length_b:
        produit = np.dot(a_np, b_np)
        convert = list(produit)
        return convert
    else:
        return [-1]
    