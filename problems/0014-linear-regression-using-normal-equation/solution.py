import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	x_np = np.array(X)
	y_np = np.array(y)
	X1 = np.dot(x_np.T, x_np)
	X2 = np.dot(x_np.T, y)
	solve = np.linalg.solve(X1, X2)
	return solve.tolist()