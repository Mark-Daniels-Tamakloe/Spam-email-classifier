import numpy as np

def logistic(w, xTr, yTr):
    """
    Compute the logistic loss and gradient.
    """
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Convert to column vector

    d, n = xTr.shape  
    wx = w.T @ xTr  # Shape: (1, n)
    ywx = yTr * wx  # Element-wise multiplication

    # Stable logistic loss calculation
    loss = np.mean(np.logaddexp(0, -ywx))

    # Compute gradient
    sigmoid = 1 / (1 + np.exp(-ywx))  # Correct sigmoid formulation
    gradient = -(xTr @ ((yTr * (1 - sigmoid)).T)) / n  # Fix gradient formulation

    return loss, gradient
