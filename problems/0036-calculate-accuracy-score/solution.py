import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code her
	y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
	y_true = np.asarray(y_true, dtype=np.float64).flatten()
	pred = np.where(y_pred>=0.5, 1, 0)
	if len(y_pred) == 0:
		return 0.0

	return float(np.mean(pred == y_true))