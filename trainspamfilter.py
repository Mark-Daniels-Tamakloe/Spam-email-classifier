
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent

def trainspamfilter(xTr,yTr):
    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!


    # (not the most successful) EXAMPLE:
    print("...using hinge loss")
    f = lambda w : logistic(w, xTr, yTr)  # Switch to logistic regression


    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-02,1500)

    # YOUR CODE HERE
    stepsize = 1e-02  # Increased step size for better convergence
    maxiter = 1500  # Increased iterations for better training

    w_trained = grdescent(f, np.zeros((xTr.shape[0], 1)), stepsize, maxiter)

    # Save trained weights
    np.save('w_trained.npy', w_trained)

    return w_trained
