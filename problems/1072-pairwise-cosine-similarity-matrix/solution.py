import numpy as np

def pairwise_cosine_similarity(X: list[list[float]]) -> list[list[float]]:
    """
    Calcule la matrice de similarité cosinus par paire pour une matrice 2D.
    
    Argument:
    X -- Une liste de listes représentant la matrice de caractéristiques (n, d).
    
    Retour:
    Une liste imbriquée (n, n) arrondie à 4 décimales.
    """
    # 1. Conversion de l'entrée en tableau NumPy de nombres flottants
    X_arr = np.array(X, dtype=float)
    
    # 2. Calcul de la norme L2 de chaque ligne (forme: (n, 1))
    # On n'ajoute PAS d'epsilon ici pour détecter précisément les lignes nulles
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
    
    # 3. Normalisation L2 des lignes avec gestion propre des divisions par zéro
    # Si la norme est 0, on remplace le dénominateur par 1 pour éviter le crash (0 / 1 = 0)
    safe_norms = np.where(norms == 0, 1.0, norms)
    X_norm = X_arr / safe_norms
    
    # 4. Produit matriciel pour obtenir toutes les similarités par paire (n, n)
    S = np.dot(X_norm, X_norm.T)
    
    # 5. Forcer la diagonale à 0.0 pour les lignes qui étaient initialement nulles
    # (np.dot sur une ligne nulle donne 0 partout, mais la consigne demande explicitement S[i,i]=0)
    is_zero_row = (norms.flatten() == 0)
    S[is_zero_row, is_zero_row] = 0.0
    
    # 6. Arrondir à 4 décimales et convertir en liste Python imbriquée
    return np.round(S, 4).tolist()
