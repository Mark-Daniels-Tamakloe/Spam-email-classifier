import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent

def trainspamfilter(xTr, yTr, loss_type="logistic", lambdaa=0.1, stepsize=1e-2, maxiter=1500):
    """
    Trains a spam filter using gradient descent with a chosen loss function.

    INPUT:
        xTr      : dxn training data matrix (each column is an input vector)
        yTr      : 1xn training label vector
        loss_type: str, one of ["logistic", "hinge", "ridge"] (default: "logistic")
        lambdaa  : float, regularization constant (only for ridge and hinge)
        stepsize : float, initial step size for gradient descent (default: 1e-2)
        maxiter  : int, maximum number of gradient descent iterations (default: 1500)

    OUTPUT:
        w_trained: dx1 trained weight vector
    """

    # Select loss function dynamically
    if loss_type == "logistic":
        print("...using logistic loss")
        f = lambda w: logistic(w, xTr, yTr)

    elif loss_type == "hinge":
        print("...using hinge loss")
        f = lambda w: hinge(w, xTr, yTr, lambdaa)

    elif loss_type == "ridge":
        print("...using ridge regression")
        f = lambda w: ridge(w, xTr, yTr, lambdaa)

    else:
        raise ValueError("Invalid loss type! Choose from 'logistic', 'hinge', or 'ridge'.")

    # Initialize weights and run gradient descent
    w_trained = grdescent(f, np.zeros((xTr.shape[0], 1)), stepsize, maxiter)

    # Save trained weights
    np.save('w_trained.npy', w_trained)

    return w_trained
