
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	count = len(y)
	count_0 = y.count(0)
	count_1 = y.count(1)
	count_2 = y.count(2)
	prob0 = count_0 / count
	prob1 = count_1 / count
	prob2 = count_2 / count
	val = 1 - ( prob0 ** 2 + prob1 ** 2 + prob2**2)
	return round(val,3)