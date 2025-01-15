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
def logistic(w,xTr,yTr):

    # YOUR CODE HERE

    return loss,gradient
