import numpy as np

def linearmodel(w, xTe):
    """
    INPUT:
    w    : d x 1 weight vector (default w=0)
    xTe  : d x n matrix (each column is an input vector)

    OUTPUT:
    preds: 1 x n vector - predictions for the input data xTe
    """
    
    # Ensure w is a column vector
    if w.ndim == 1:
        w = w.reshape(-1, 1)  # Convert to column vector (d x 1)
    
    # Compute predictions as w^T * xTe
    preds = (w.T @ xTe).flatten()  # Ensure output is a 1D array (1 x n)
    
    return preds
