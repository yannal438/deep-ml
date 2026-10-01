import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N, D = X.shape
    
    # Initialisation des paramètres
    weights = np.zeros(D)
    bias = 0.0
    loss_history = []
    
    for i in range(iterations):
        # 1. Calcul linéaire et activation
        z = X @ weights + bias
        sigmoide = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        # 2. Calcul des gradients sous forme de SOMME (sans division par N)
        grad_weights = X.T @ (sigmoide - y)
        grad_bias = np.sum(sigmoide - y)

        # 3. Mises à jour des paramètres
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        # 4. Calcul de la perte BCE totale (Somme)
        part1 = y @ np.log(sigmoide + 1e-15)
        parti2 = (1 - y) @ np.log(1 - sigmoide + 1e-15)
        current_loss = -(part1 + parti2)
        loss_history.append(float(current_loss))
        
    # --- DEHORS DE LA BOUCLE ---
    # A. Combiner avec le biais EN PREMIER pour correspondre à l'ordre du test
    coefficients_optimises = np.round(np.append(bias, weights), 4).tolist()
    
    # B. Arrondir toutes les pertes à la 4e décimale
    pertes_historique = [round(loss, 4) for loss in loss_history]
    
    return coefficients_optimises, pertes_historique

