import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.asarray(X).squeeze()
    
    # Sécurité si X n'a qu'une seule ligne
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    n_samples, n_features = X.shape
    n_polynomial = []

    for d in range(degree + 1):
        for combos in combinations_with_replacement(range(n_features), d):
            if d == 0:
                # ✅ CORRECTION 1 : On crée un tableau 1D de taille (n_samples,)
                col = np.ones(n_samples)
            else:
                # np.prod avec axis=1 renvoie déjà un tableau 1D de taille (n_samples,)
                col = np.prod(X[:, combos], axis=1)
                
            n_polynomial.append(col)
            
    # ✅ CORRECTION 2 : Retrait des crochets inutiles autour de n_polynomial
    res = np.vstack(n_polynomial).T
    return np.sort(res, axis=1)
