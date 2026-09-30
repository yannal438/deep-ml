import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col is None:
		n_col = np.max(x) + 1
	row = x.shape[0]
	x_np = np.zeros((row, n_col), dtype = float)
	x_np[np.arange(row), x] = 1
	return x_np.tolist()
