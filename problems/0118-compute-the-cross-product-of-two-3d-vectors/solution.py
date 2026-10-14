import numpy as np

def cross_product(a, b):
    # Ensure both inputs have exactly 3 dimensions
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Both input vectors must be exactly 3-dimensional.")
        
    w_x = a[1] * b[2] - a[2] * b[1]
    w_y = a[2] * b[0] - a[0] * b[2]
    w_z = a[0] * b[1] - a[1] * b[0]
    
    return [w_x, w_y, w_z]
