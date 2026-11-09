import numpy as np
import random

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	logits_np = np.array(logits)
	logits_stable = logits_np - np.max(logits_np, axis=-1, keepdims=True)
	logits_exp = np.exp(logits_stable)

	soft = logits_exp /np.sum(logits_exp, axis=-1, keepdims=True)
	
	gradient = soft.copy()
	gradient[target] -=1
	return gradient.tolist()
	
	