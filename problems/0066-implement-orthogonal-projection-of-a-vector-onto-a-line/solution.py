import numpy as np
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	num =  np.dot(v, L)
	denom = np.linalg.norm(L)
	deno = denom ** 2
	scalar = num / deno
	p = np.dot(scalar, L)
	return p