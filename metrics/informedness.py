from collections import OrderedDict
import numpy as np
from sklearn.metrics import confusion_matrix

def informedness(y_true, y_pred):
    """
    Compute the informedness value.
    
    Informedness returns the probability that a model is making informed decisions.
    The best value is 1 and the worst value is 0. Negative values indicate informedness
    with reversed labels.
    
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    
    Returns
    -------
    informedness : float
        Informedness score.
        
    Notes
    -----
    Informedness is also known as Bookmaker Informedness.
    
    References
    ----------
    .. [1] Powers, David M.W. (2003).
           Recall and Precision versus the Bookmaker.
           International Conference on Cognitive Science.
    
    Examples
    --------
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> informedness(y_true, y_pred)
    0.625
    """
    # Build an explicit mapping from labels to idx
    all_labels = set(y_true).union(set(y_pred))
    
    idx2l = OrderedDict((idx, l) for idx, l in enumerate(sorted(all_labels)))
    l2idx = {k: v for v, k in idx2l.items()}

    cm = confusion_matrix(y_true, y_pred, labels=list(idx2l.values()))
    N = np.sum(cm)

    # rprob = real prob = empirical prob of class
    rprob = np.sum(cm, axis=0) / N
    # pprob = prediction prob 
    pprob = np.sum(cm, axis=1) / N

    BM_gain = 0  # Bookmaker gain
    
    # Calculate Bookmaker gain
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            g = 1 / rprob[l2idx[yt]]
        else:
            g = -1 / (1 - rprob[l2idx[yt]])
        BM_gain += g * pprob[l2idx[yt]] / N

    return BM_gain