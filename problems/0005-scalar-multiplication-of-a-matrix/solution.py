import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	matrix_np = np.array(matrix)
	scalar_np = np.array(scalar)

	return (matrix_np * scalar_np).tolist()
	pass
