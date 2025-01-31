import numpy as np

def logistic(w, xTr, yTr):
    """
    Compute the logistic loss and gradient.
    """
    # Ensure correct shape of w (d x 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Convert to column vector

    # Get dimensions
    d, n = xTr.shape  

    # Compute predictions
    wx = w.T @ xTr  # Shape: (1, n)
    ywx = yTr * wx  # Element-wise multiplication (1, n)

    # Compute logistic loss
    loss = np.sum(np.log(1 + np.exp(-ywx))) / n  # Normalize by n

    # Compute gradient (fix sign)
    sigmoid = 1 / (1 + np.exp(-ywx))  # Corrected sigmoid function
    gradient = (xTr @ ((sigmoid - 1) * yTr).T) / n  # Corrected gradient formula

    return loss, gradient
