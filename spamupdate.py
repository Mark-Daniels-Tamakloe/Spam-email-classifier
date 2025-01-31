def spamupdate(w, email, truth, eta=0.9):
    """
    Update the weight vector when the classifier makes a mistake.

    INPUT:
    w      : dx1 weight vector
    email  : dx1 email feature vector
    truth  : scalar (true label, either +1 or -1)
    eta    : float (learning rate, default 0.01)

    OUTPUT:
    w_updated : dx1 updated weight vector
    """

    # Compute weight update
    w_updated = w + eta * truth * email.reshape(-1, 1)

    # Debugging: Print the first few updated weights
    #print(f"Updated weights (first 5): {w_updated[:5].flatten()}")

    return w_updated
