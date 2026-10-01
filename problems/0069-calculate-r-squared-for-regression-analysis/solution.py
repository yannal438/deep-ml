import numpy as np

def r_squared(y_true, y_pred):
    
    somme = (y_true - y_pred)** 2
    somme_carre_residu = np.sum(somme, axis=0)

    somme_1 = (y_true - np.mean(y_true, axis=0))**2
    somme_carre_totale = np.sum(somme_1, axis = 0)
    r2 = 1 - (somme_carre_residu/somme_carre_totale)
    return r2
