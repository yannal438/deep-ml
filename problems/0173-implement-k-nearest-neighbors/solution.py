import numpy as np
import math
def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points_np = np.array(points)
    query = np.array(query_point)

    # Calculons la disatance
    distance = np.sqrt(np.sum((points_np - query) ** 2, axis=1))
    
    # Trier les indices du proches au plus éloignés
    indices_tries = np.argsort(distance, kind='stable')
    # Selectionnons les K premiers points et les convertir en tuple(point)
    point_proches = points_np[indices_tries[:k]]
    return  [tuple(p) for p in point_proches]