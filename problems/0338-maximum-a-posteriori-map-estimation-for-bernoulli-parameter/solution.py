import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    nombre_succes = sum(observations)
    nombre_echec = len(observations) - nombre_succes
    alpha_prime= alpha + nombre_succes
    beta_prime = beta + nombre_echec

    if (alpha_prime + beta_prime - 2) == 0:
        return 0.5
    theta_map = (alpha_prime - 1) / (alpha_prime + beta_prime - 2)

    theta_map = np.clip(theta_map, 0.0, 1.0)
    return round(theta_map, 4)

