import math

def sigmoid(z: float) -> float:
	#Your code here
	sigmoide = 1 / (1 + math.exp(-z))
	return float(sigmoide)