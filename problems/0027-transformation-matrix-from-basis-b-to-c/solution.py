import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B_np = np.array(B)
	C_np = np.array(C)
	# Verfions si C est inversible
	determinant = np.linalg.det(C)
	if determinant == 0:
		return []
	else:
		P = np.linalg.inv(C) @ B_np
		return P.tolist()