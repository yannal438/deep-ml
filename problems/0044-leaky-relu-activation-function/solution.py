def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	# Your code here
	import numpy as np
	return np.where(z<0, alpha * z, z)
