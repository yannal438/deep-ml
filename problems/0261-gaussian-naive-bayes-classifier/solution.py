import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)
    
    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    # 1. Préparation des données d'entraînement
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    
    prioris = {}
    moyenne = {}
    variance = {}

    # 2. Phase d'entraînement (Calcul des paramètres par classe)
    for c in classes:
        # Sélectionner uniquement les lignes de la classe c
        X_c = X_train[y_train == c]

        # Stocker les probabilités a priori, moyennes et variances
        prioris[c] = X_c.shape[0] / n_samples
        moyenne[c] = np.mean(X_c, axis=0)
        variance[c] = np.var(X_c, axis=0) + 1e-9

    # 3. Phase de prédiction (Pour chaque échantillon de test)
    predictions = []

    for x in X_test:
        posteriors = {} # Dictionnaire temporaire pour stocker le score de chaque classe
        
        for c in classes:
            log_prioris = np.log(prioris[c])
            mu = moyenne[c]
            var = variance[c]
            
            # Formule de la cloche de Gauss au format Logarithme
            log_vrais = -0.5 * np.log(2 * np.pi * var) - ((x - mu) ** 2) / (2 * var)
            
            # Théorème de Bayes : Log(Prior) + Somme des Log(Vraisemblances)
            posteriors[c] = log_prioris + np.sum(log_vrais)
            
        # Sélectionner la classe qui a obtenu le score maximal
        classe_n = max(posteriors, key=posteriors.get)
        predictions.append(classe_n)

    return np.array(predictions)
