import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    pass
    X = np.asarray(X).squeeze()
    n_samples, n_features = X.shape 
    n_polynomial = []
    if X.ndim == 1:
        X = X.reshape(1, -1)

    for i in range(degree + 1):
        for combos in combinations_with_replacement(range(n_features), i):
            if i == 0:
                col = np.ones(n_samples)
            else:
                col = np.prod(X[:, combos], axis = 1)
            n_polynomial.append(col)
    res = np.vstack(n_polynomial).T
    return np.sort(res, axis =1)
            