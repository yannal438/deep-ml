import numpy as np
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	x = np.asarray(features)
	y = np.asarray(labels)
	w = np.asarray(initial_weights)
	b = initial_bias

	mse_li = []
	n = len(x)

	for _ in range(epochs):
		# Forward
		z = np.dot(x, w) + b 
		s = sigmoid(z)

		# Loss MSE 
		erreur = s - y 
		MSE = 1/n * (np.sum(erreur**2))
		mse_li.append(round(MSE, 4))

		# Backward
		s_prime = s * (1 - s)
		gradient = (2 /n) * erreur*s_prime
		gradient_w = x.T @ gradient

		grad = (2/n) *erreur * s_prime
		gradient_b = grad.sum()

		# Mise à jour des parametres
		w -= learning_rate * gradient_w
		b -= learning_rate * gradient_b



	return w, b, mse_li