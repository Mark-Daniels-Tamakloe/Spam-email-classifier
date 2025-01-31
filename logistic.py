import numpy as np

def logistic(w, xTr, yTr):
    """
    Compute the logistic loss and gradient.
    """
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Convert to column vector
    
    if yTr.shape[0] > 1:
        yTr = yTr.reshape(1, -1)  # Ensure yTr is a row vector

    d, n = xTr.shape  

    wx = w.T @ xTr  # Shape: (1, n)
    ywx = yTr * wx  # Element-wise multiplication (1, n)

    # Compute logistic loss (numerically stable version)
    loss = np.mean(np.logaddexp(0, -ywx))  

    # Compute gradient (correct formula)
    sigmoid = 1 / (1 + np.exp(-ywx))  
    gradient = -(xTr @ ((yTr * (1 - sigmoid)).T)) / n  

    return loss, gradient
