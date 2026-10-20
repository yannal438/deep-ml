import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing
        self.prioris = {}
        self.prob_features = {}
        self.classes = []

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            # Selection des lignes correspondantes a la classe classe C 
            X_c = X[y == c]

            # Calcul de la probabilité a prioris de la classe c 
            self.prioris[c] = X_c.shape[0] / n_samples

            # On compte le nombre de fois chaque caracteristique est a 1 dans cette classe
            total_compte_1 = np.sum(X_c, axis=0)

            # Formule du lissage
            self.prob_features[c] = (total_compte_1 + self.smoothing) / (X_c.shape[0] + 2 * self.smoothing)

    def predict(self, X):
        # Predict class labels for test set X
        X = np.array(X)
        predictions = [self._predire(x) for x in X]

        return np.array(predictions)

    def _predire(self, x):
        posteriors = {}
        for c in self.classes:
            log_prob_classe = np.log(self.prioris[c])

            # Recuperer les probabilité des caracteristiques
            p_features_c = self.prob_features[c]
            # Formule de Bernoulli : 
            # Si x_i == 1 -> on prend log(p)
            # Si x_i == 0 -> on prend log(1 - p)
            log_prob_features = x * np.log(p_features_c) + (1 - x) * np.log(1 - p_features_c)
            
            # Somme des probabilités log pour cette classe
            posteriors[c] = log_prob_classe + np.sum(log_prob_features)
            
        # On retourne la classe qui a la probabilité la plus élevée
        return max(posteriors, key=posteriors.get)







