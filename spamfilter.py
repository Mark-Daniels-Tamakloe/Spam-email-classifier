import numpy as np
from linearmodel import linearmodel
from area_under_roc_curve import area_under_roc_curve
from spamupdate import spamupdate

def spamfilter(xTe, yTe, w_trained, thresh=0.05):
    """
    Spam filter using linear classification.

    INPUT:
    xTe      : dxn data matrix (test set)
    yTe      : 1xn label matrix (ground truth labels)
    w_trained: dx1 weight vector (trained model)
    thresh   : float - classification threshold (default: 0.3)

    OUTPUT:
    fpr : False positive rate
    tpr : True positive rate
    auc : Area under the ROC curve
    """

    [d, n] = np.shape(xTe)

    # Initialize counters
    fpr = 0
    tpr = 0
    allpreds = np.zeros((1, n))

    for i in range(n):
        email = xTe[:, i].reshape(-1, 1)  # Ensure correct shape
        truth = yTe[:, i]

        # Compute raw prediction
        rawpred = linearmodel(w_trained, email).item()  # Ensure it's a scalar

        # Apply threshold for classification
        pred = 1 if rawpred > thresh else -1

        # Count false positives and true positives
        if pred != truth:
            if pred == 1:  # False positive (classified as spam but not spam)
                fpr += 1

            # Apply spam update if wrong prediction
            w_trained = spamupdate(w_trained, email, truth)

        else:  # Correct classification
            if pred == 1:  # True positive (correctly classified spam)
                tpr += 1

        # Store predictions
        allpreds[:, i] = rawpred

    # Compute ROC curve metrics
    a, b, auc = area_under_roc_curve(yTe, allpreds)

    # Compute FPR and TPR correctly
    total_positives = np.sum(yTe == 1)
    total_negatives = np.sum(yTe == -1)

    tpr = tpr / total_positives if total_positives > 0 else 0
    fpr = fpr / total_negatives if total_negatives > 0 else 0

    # Print final evaluation metrics
    print("False positive rate: {:.2f}%".format(fpr * 100))
    print("True positive rate: {:.2f}%".format(tpr * 100))
    print("AUC: {:.2f}%".format(float(np.mean(auc)) * 100))

    return a, b, auc
