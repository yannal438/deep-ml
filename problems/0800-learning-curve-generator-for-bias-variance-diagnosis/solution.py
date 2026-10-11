import numpy as np

def polynomial(X, degree):
    X_flat = np.asarray(X).flatten()
    X_poly = np.vstack([X_flat**i for i in range(degree+1)]).T
    return X_poly

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).flatten()
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).flatten()
    coeff = polynomial(X_val, degree)
    train_errors = []
    val_errors = []
    for n in train_sizes:
        X_subset = X_train[:n]
        y_subset = y_train[:n]
        X_subset_train = polynomial(X_subset, degree)
        w = np.linalg.pinv(X_subset_train.T @ X_subset_train) @ (X_subset_train.T @ y_subset)
        train_predit = np.dot(X_subset_train, w)
        val_predit = np.dot(coeff, w)
        mse_train = np.mean((train_predit - y_subset)**2)
        mse_val = np.mean((val_predit - y_val)**2)
        train_errors.append(mse_train)
        val_errors.append(mse_val)
    final_train = train_errors[-1]
    final_val = val_errors[-1]

    if (final_train > bias_threshold):
        diagnostic ='high_bias'

    elif (final_val - final_train) > variance_threshold:
        diagnostic = 'high_variance'
    else:
        diagnostic = 'good_fit'
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': diagnostic
    }

