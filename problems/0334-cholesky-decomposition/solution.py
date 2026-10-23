import numpy as np


def cholesky_decomposition(A):
  """Perform Cholesky decomposition on a symmetric positive-definite matrix.

  Args:
      A: A symmetric positive-definite matrix (2D list or numpy array)

  Returns:
      L: Lower triangular matrix such that A = L @ L.T as a 2D list, or -1 if
      decomposition is not possible
  """
  try:
    if A is None:
      return -1

    # Conversion forcée en tableau NumPy de floats pour s'assurer du type
    A_np = np.array(A, dtype=float)

    # CORRIGÉ : ndim au lieu de dim
    if A_np.size == 0 or A_np.ndim != 2:
      return -1

    lignes, colonnes = A_np.shape
    if lignes != colonnes:
      return -1

    if not np.allclose(A_np, A_np.T, atol=1e-8):
      return -1

    L_np = np.linalg.cholesky(A_np)

    # CORRIGÉ : L_np.tolist() au lieu de L.np.tolist()
    return L_np.tolist()

  # CORRIGÉ : Casse respectée pour ValueError et LinAlgError
  except (ValueError, TypeError, np.linalg.LinAlgError):
    return -1
