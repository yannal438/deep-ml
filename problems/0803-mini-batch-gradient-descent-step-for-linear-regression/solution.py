import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    # 1. Sélectionner le mini-batch à l'aide des indices fournis
    X_batch = X[batch_indices]       # Forme: (m, D)
    y_batch = y[batch_indices]       # Forme: (m,)
    m = len(batch_indices)           # Taille du mini-batch
    
    # 2. Calculer les prédictions du modèle pour ce batch
    # X_batch @ weights donne un vecteur de taille (m,), auquel on ajoute le biais
    predictions = X_batch @ weights + bias
    
    # 3. Calculer l'erreur (différence entre prédictions et valeurs réelles)
    error = predictions - y_batch    # Forme: (m,)
    
    # 4. Calculer les gradients par rapport aux poids et au biais
    # Pour les poids : (2/m) * X_batch^T @ error
    gradient_w = (2 / m) * (X_batch.T @ error)
    # Pour le biais : (2/m) * somme(error)
    gradient_b = (2 / m) * np.sum(error)
    
    # 5. Mettre à jour les paramètres avec le learning rate (lr)
    updated_weights = weights - lr * gradient_w
    updated_bias = bias - lr * gradient_b
    
    # 6. Combiner les poids mis à jour et le biais mis à jour dans un tableau unique de taille (D+1,)
    return np.append(updated_weights, updated_bias)


