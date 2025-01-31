from numpy import maximum
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
def hinge(w, xTr, yTr, lambdaa):
    """
    Compute the hinge loss and gradient.
    """
    # Ensure correct shape of w (d x 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Convert to column vector

    # Compute hinge loss: max(0, 1 - y_i * (w^T x_i))
    margins = 1 - yTr * (w.T @ xTr)  # (1, n) array
    losses = np.maximum(0, margins)  # Hinge loss (1, n)

    # Compute total loss (hinge loss + regularization)
    hinge_loss = np.sum(losses)  # Scalar hinge loss
    reg_loss = hinge_loss + lambdaa * np.sum(w ** 2)  # Ridge regularization term

    # Compute the gradient
    indicator = (margins > 0).astype(float)  # Indicator function (1, n)
    gradient = -xTr @ (indicator * yTr).T + 2 * lambdaa * w  # (d, 1)

    return reg_loss, gradient
