import numpy as np

def logistic(w, xTr, yTr):
    """
    Compute the logistic loss and gradient.
    """
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Ensure w is a column vector
    
    if yTr.shape[0] > 1:
        yTr = yTr.reshape(1, -1)  # Ensure yTr is a row vector

    d, n = xTr.shape  

    wx = w.T @ xTr  # Shape: (1, n)
    ywx = yTr * wx  # Element-wise multiplication (1, n)

    # Compute logistic loss
    loss = np.sum(np.log(1 + np.exp(-ywx)))  

    # Compute gradient
    sigmoid = 1 / (1 + np.exp(-ywx))  
    gradient = -(xTr @ ((np.exp(-ywx)*sigmoid) * yTr).T)  # Fix gradient direction

    print(f" gradient shape: {gradient.shape}, w shape: {w.shape}")

    return loss, gradient
