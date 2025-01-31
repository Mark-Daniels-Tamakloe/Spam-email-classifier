import numpy as np

def grdescent(func, w0, stepsize, maxiter, tolerance=1e-2):
    """
    Gradient descent with adaptive step size.

    INPUT:
        func      : function to minimize (returns loss and gradient)
        w0        : initial weight vector
        stepsize  : initial gradient descent step size
        maxiter   : maximum number of iterations
        tolerance : stopping criterion based on gradient norm

    OUTPUT:
        w : final optimized weight vector
    """
    eps = 2.2204e-14  # Minimum step size
    max_step = 1e1    # Maximum step size to avoid divergence

    w = w0.copy()  # Copy initial weights to avoid modifying original
    prev_loss, _ = func(w)  # Compute initial loss

    for i in range(maxiter):
        # Compute loss and gradient
        loss, gradient = func(w)

        # If gradient norm is small, stop
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < tolerance:
            break

        # Update weights using gradient descent
        w_new = w - stepsize * gradient  # Gradient step

        # Compute new loss
        new_loss, _ = func(w_new)

        # Adjust step size based on loss behavior
        if new_loss < loss:  # Loss decreased → Increase step size
            stepsize = min(stepsize * 1.01, max_step)
            w = w_new  # Accept new weights
        else:  # Loss increased → Decrease step size
            stepsize = max(stepsize * 0.5, eps)

    return w
