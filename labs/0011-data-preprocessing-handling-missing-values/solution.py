import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    # On fait une copie du tableau
    X_clean = X.copy()
    # On recupere le nombre de ligne et de colonne
    nb_ligne, nb_colonne = X_clean.shape

    # On va creer ue boucle pour parcourir chaque colonne une par une
    for i in range(nb_colonne):
        # On extrait toute la colonne i
        colonne = X_clean[:, i]
        # Trouver les indices ou les valeurs sont manquantes
        manq = np.isnan(colonne)
        # Extraire uniquement les valeurs valides
        valeurs_valide = colonne[~manq]
        # Calculer la mediane des valeurs valides
        valeur_r = np.median(valeurs_valide)

        # Remplacer les NaN de cette colonne par la mediane calculé
        X_clean[manq, i] = valeur_r

    
    # TODO: Fill in NaN values
    
    return X_clean
