import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	# Fonction sigmoide
	sigmoide = 1 / (1 + np.exp(-x))
	s = sigmoide
	derivate_s = s * (1.0 - s)

	# Fonction tangente hyperbolique
	tanh = np.tanh(x)
	t = tanh
	derivate_t = 1 - t**2

	# Fonction ReLu
	relu = np.where(x<=0, 0.0, 1.0)

	return {
		'sigmoid' : float(derivate_s),
		'tanh' : float(derivate_t),
		'relu' : float(relu)
	}