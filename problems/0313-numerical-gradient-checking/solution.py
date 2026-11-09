import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    x = np.array(x, dtype=float)
    analytical_grad = np.array(analytical_grad, dtype= float)
    numerical_gradient = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        numerical_gradient[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    num_error = np.linalg.norm(numerical_gradient - analytical_grad)
    deno_error = np.linalg.norm(numerical_gradient) + np.linalg.norm(analytical_grad)

    if deno_error < 1e-5:
        relative_error = 0.0
    else:
        relative_error = num_error / deno_error
    return numerical_gradient, relative_error
        