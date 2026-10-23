import numpy as np


def lu_decomposition(A: list) -> tuple:
  """Decomposes a square matrix A into a lower triangular matrix L and an upper

  triangular matrix U using Doolittle's algorithm without pivoting.

  Args:
      A: Square matrix as a list of lists

  Returns:
      tuple: (L, U) where L and U are numpy arrays, L has 1s on the diagonal,
             and A = L @ U.
  """
  A_np = np.array(A, dtype=float)
  n = A_np.shape[0]

  # Initialize L as identity matrix (1s on diagonal) and U as a copy of A
  L = np.eye(n)
  U = A_np.copy()

  # Doolittle algorithm without pivoting
  for k in range(n - 1):
    # Eliminate elements below the pivot entry U[k, k]
    for i in range(k + 1, n):
      # Calculate the multiplier factor
      facteur = U[i, k] / U[k, k]

      # Store the factor in the lower triangular matrix L
      L[i, k] = facteur

      # Update the remaining elements of row i in the upper matrix U
      U[i, k:] -= facteur * U[k, k:]

  return L, U

