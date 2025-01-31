import math
import numpy as np
'''
    INPUT:
    xTr: dxn matrix - 2d numpy array (each column is an input vector)
    yTr: 1xn vector - 2d numpy array (each entry is a label)
    w :  dx1 weight vector - 2d numpy array (default w=0)

    OUTPUTS:

    loss:      float (the total loss obtained with w on xTr and yTr)
    gradient:  dx1 vector - 2d numpy array (the gradient at w)

    [d,n]=size(xTr);
'''
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

    # Compute gradient
    sigmoid = 1 / (1 + np.exp(ywx))  # Sigmoid function
    gradient = - (xTr @ (yTr * sigmoid).T) / n  # Normalize by n

    return loss, gradient