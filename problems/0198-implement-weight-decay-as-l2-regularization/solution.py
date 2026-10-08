import numpy as np

def apply_weight_decay(
    parameters: list[list[float]], 
    gradients: list[list[float]], 
    lr: float, 
    weight_decay: float, 
    apply_to_all: list[bool]
) -> list[list[float]]:
    
    liste = []
    for param_group, grad_group, should_decay in zip(parameters, gradients, apply_to_all):
        p = np.array(param_group, dtype=float)
        n = np.array(grad_group, dtype=float)
        
        if should_decay:
            lists = p - lr * (n + weight_decay * p)
        else:
            lists = p - lr * n 
        liste.append(lists.tolist())
        
    return liste
