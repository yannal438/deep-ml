import numpy as np
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	a_np = np.array(a)
	b_np = np.array(b)
	if len(a_np) != len(b_np):
		return -1
	else:
		return np.sum([a_np, b_np], axis = 0).tolist()