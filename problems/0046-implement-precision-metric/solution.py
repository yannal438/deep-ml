import numpy as np

def precision(y_true, y_pred) -> float:
    """
    Calculate the precision score from scratch using NumPy.
    """
    # 1. Conversion propre en tableaux NumPy 1D
    y_true_np = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred_np = np.asarray(y_pred, dtype=np.float64).flatten()
    
    # 2. Vérification des dimensions pour éviter le -1.0
    if y_true_np.shape != y_pred_np.shape:
        return 0.0
        
    # 3. Binarisation (0 ou 1) avec le seuil à 0.5
    y_bin = np.where(y_pred_np >= 0.5, 1, 0)

    # 4. Calcul exact des Vrais Positifs (VP) et Faux Positifs (FP)
    vp = int(np.sum((y_bin == 1) & (y_true_np == 1)))
    fp = int(np.sum((y_bin == 1) & (y_true_np == 0)))
    
    # 5. Sécurité contre la division par zéro
    if (vp + fp) == 0:
        return 0.0
        
    # 6. Retourne la précision
    return float(vp / (vp + fp))
