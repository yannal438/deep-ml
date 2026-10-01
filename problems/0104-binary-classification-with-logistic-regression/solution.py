import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.
	
	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	X = np.asarray(X, dtype = np.float64)
	weights = np.asarray(weights, dtype = np.float64).flatten()
	z  = X @ weights + bias
	sigmoid = 1 / (1 + np.exp(-z))
	return np.where(sigmoid >= 0.5, 1, 0)
	
    