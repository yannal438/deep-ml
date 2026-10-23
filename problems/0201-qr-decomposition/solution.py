import numpy as np


def qr_decomposition(
    A: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]] | int:
  """Perform QR decomposition using Gram-Schmidt process.

  Args:
      A: An m x n matrix represented as list of lists

  Returns:
      Tuple of (Q, R) where Q is orthogonal and R is upper triangular as lists,
      or -1 if the input is invalid or columns are linearly dependent.
  """
  try:
    if A is None:
      return -1
    A_np = np.array(A, dtype=float)
    if A_np.size == 0 or A_np.ndim != 2:
      return -1

    lignes, colonnes = A_np.shape
    Q = np.zeros((lignes, colonnes))

    R = np.zeros((colonnes, colonnes))

    for j in range(colonnes):

      v_j = A_np[:, j].copy()


      for i in range(j):

        R[i, j] = np.dot(A_np[:, j], Q[:, i])
        v_j -= R[i, j] * Q[:, i]

      r_jj = np.linalg.norm(v_j)

      if r_jj < 1e-12:
        return -1

      R[j, j] = r_jj
      Q[:, j] = v_j / r_jj


    return Q.tolist(), R.tolist()

  except (ValueError, TypeError):
    return -1
