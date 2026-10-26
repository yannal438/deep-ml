import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	num_stable = x - np.max(x)
	soft_num = np.exp(num_stable)
	soft_de = np.sum(soft_num)
	softmax = soft_num / soft_de

	J = np.diag(softmax) - np.outer(softmax, softmax)
	return J