import numpy as np

def ridge(w, xTr, yTr, lambdaa):
    """
    Compute the ridge regression loss and gradient.
    """
    if w.ndim == 1:
        w = w.reshape(-1, 1)  

    d, n = xTr.shape  

    error = (w.T @ xTr) - yTr  

    # Compute ridge regression loss (normalized MSE + regularization)
    loss = np.sum(error ** 2) + lambdaa * np.sum(w ** 2)

    # Compute gradient
    gradient = (2) * (xTr @ error.T) + 2 * lambdaa * w  

    return loss, gradient
