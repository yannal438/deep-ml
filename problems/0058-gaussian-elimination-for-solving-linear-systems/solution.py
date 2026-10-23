import numpy as np


def gaussian_elimination(A, b):
  """Solves the system Ax = b using Gaussian Elimination with partial pivoting.

  :param A: Coefficient matrix
  :param b: Right-hand side vector
  :return: Solution vector x
  """
  A = np.array(A, dtype=float)
  b = np.array(b, dtype=float)
  n = len(b)

  # Phase 1: Elimination directe (Descente)
  for k in range(n - 1):
    # On cherche l'element le plus grand en valeur absolue dans la colonne k
    ligne_pivot = k + np.argmax(np.abs(A[k:n, k]))

    # CORRIGÉ : Utilisation de '=' au lieu de '==' pour l'échange de lignes
    if ligne_pivot != k:
      A[[k, ligne_pivot]] = A[[ligne_pivot, k]]
      b[[k, ligne_pivot]] = b[[ligne_pivot, k]]

    # Sécurité numérique : évite la division par zéro si la matrice est singulière
    if np.abs(A[k, k]) < 1e-12:
      raise np.linalg.LinAlgError("Pivot nul ou trop proche de zéro.")

    # CORRIGÉ : Désindenté du 'if' pour s'exécuter à chaque étape k
    for i in range(k + 1, n):
      facteur = A[i, k] / A[k, k]
      # CORRIGÉ : Slicing 'k:' pour correspondre à la taille du vecteur soustrait
      A[i, k:] -= facteur * A[k, k:]
      b[i] -= facteur * b[k]

  # Phase 2: Substitution inverse (Remontée)
  X = np.zeros(n)
  X[n - 1] = b[n - 1] / A[n - 1, n - 1]

  for i in range(n - 2, -1, -1):
    produit_scalaire = np.dot(A[i, i + 1 :], X[i + 1 :])
    X[i] = (b[i] - produit_scalaire) / A[i, i]

  return X

