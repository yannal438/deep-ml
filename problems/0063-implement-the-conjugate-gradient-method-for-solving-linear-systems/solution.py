import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	A = np.array(A, dtype=float)
	b = np.array(b, dtype=float)
	# Initialisation
	if x0 is None:
		x = np.zeros_like(b)
	else:
		x = np.array(b)
	# Residu initial
	r = b - np.dot(A, x)
	p = r.copy() # Direction de recherche initiale

	num = np.dot(r, r)
	for k in range(n):
		if np.sqrt(num) < tol:
			break
		Ap = np.dot(A, p)
		alpha = num / np.dot(p, Ap)
		# Mise a jour des solutions
		x = x + alpha * p 
		r_new = r - alpha * Ap 
		carre_norm_r = np.dot(r_new, r_new)
		if np.sqrt(carre_norm_r) < tol:
			x = x
			break
		beta = carre_norm_r / num
		p = r_new + beta * p 
		r = r_new
		num = carre_norm_r
	return x.tolist()

