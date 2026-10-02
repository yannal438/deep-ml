import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places

	"""
	y_true = np.asarray(y_true, dtype = np.float64).flatten()
	y_pred = np.asarray(y_pred, dtype = np.float64).flatten()
	y_pred_bin = np.where(y_pred>=0.5, 1, 0)
	if y_pred.shape != y_true.shape:
		return 0.0

	vp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
	fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
	vn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
	fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))
	
	num = (1 + (beta)** 2)
	den = (num * vp + fp + (beta**2) * fn)
	f2_beta = num * vp / den
	return np.round((f2_beta), 3)

	
