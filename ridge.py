
import numpy as np
'''
    INPUT:
    xTr:    dxn matrix - 2d numpy array (each column is an input vector)
    yTr:    1xn vector - 2d numpy array (each entry is a label)
   lambdaa: float (regularization constant)
    w:      dx1 weight vector - 2d numpy array (default w=0)

    OUTPUTS:

    reg_loss:      float (the total regularized loss obtained with w on xTr and yTr)
    gradient:  dx1 vector - 2d numpy array (the gradient at w)

    [d,n]=size(xTr);
'''

def ridge(w, xTr, yTr, lambdaa):
    """
    Compute the ridge regression loss and gradient.
    """
    # Ensure correct shape of w (d x 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Convert to column vector

    # Get dimensions
    d, n = xTr.shape  

    # Compute squared loss
    error = (w.T @ xTr) - yTr  # (1 x n) - (1 x n) → (1 x n)
    
    # Compute ridge regression loss
    loss = np.sum(error ** 2) / n  # Mean squared error
    reg_term = lambdaa * np.sum(w ** 2)  # Regularization term
    reg_loss = loss + reg_term

    # Compute gradient
    gradient = (2 / n) * (xTr @ error.T) + 2 * lambdaa * w  # (d x 1)

    return reg_loss, gradient