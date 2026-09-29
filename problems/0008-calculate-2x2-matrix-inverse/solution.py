import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
	# Your code here
	matrix_np = np.array(matrix)
	determinant = np.linalg.det(matrix)
	# Verifions si la matrice est inversible
	if determinant == 0:
		return None
	else:
		return (np.linalg.inv(matrix_np)).tolist()