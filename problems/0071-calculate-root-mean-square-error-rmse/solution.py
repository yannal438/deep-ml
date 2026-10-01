import numpy as np

def rmse(y_true, y_pred):
    # 1. Sécurité du type : Conversion obligatoire en tableaux NumPy numériques
    try:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
    except (ValueError, TypeError):
        return -1.0

    # 2. Sécurité des dimensions : Aplatir pour éviter les pièges du broadcasting
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 3. Vérification de la compatibilité des tailles
    if y_true_flat.shape[0] != y_pred_flat.shape[0]:
        return -1.0
        
    # 4. Vérification si les tableaux sont vides
    n = len(y_true_flat)
    if n == 0:
        return 0.0
        
    # 5. Calcul de la RMSE
    somme = (y_true_flat - y_pred_flat) ** 2
    rmse_re = np.sum(somme) / n
    rmse_res = np.sqrt(rmse_re)
    
    return round(rmse_res, 3)
