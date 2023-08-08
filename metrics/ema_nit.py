import numpy as np
from sklearn.metrics import confusion_matrix

def shannon_entropy(X):
    """
    Calculate Shannon entropy of a probability distribution.
    
    Parameters:
    - X: array-like, unique events
    
    Returns:
    - H: float, Shannon entropy
    """
    _, num_unique_events = np.unique(X, return_counts=True, axis=0)
    probs = num_unique_events / len(X)
    H = -np.sum(probs * np.log2(probs))
    return H
    
def joint_shannon_entropy(Y, X):
    """
    Calculate joint Shannon entropy of two arrays Y and X.
    
    Parameters:
    - Y: array-like, first event
    - X: array-like, second event
    
    Returns:
    - H: float, joint Shannon entropy
    """
    YX = np.c_[Y, X]
    return shannon_entropy(YX)

def conditional_shannon_entropy(Y, X):
    """
    Calculate conditional Shannon entropy H(Y|X) = H(X;Y) - H(X).
    
    Parameters:
    - Y: array-like, conditioning event
    - X: array-like, conditioned event
    
    Returns:
    - H: float, conditional Shannon entropy
    """
    return joint_shannon_entropy(X, Y) - shannon_entropy(X)

def normalised_information_transfer(y_true, y_pred):
    """
    Calculate Normalized Information Transfer (NIT) between true and predicted labels.
    NIT is defined in Valverde-Albacete and Peláez-Moreno, 2014

    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels
    
    Returns:
    - NIT: float, Normalized Information Transfer
    """
    H_X = shannon_entropy(y_true)
    H_Y = shannon_entropy(y_pred)
    H_XY = joint_shannon_entropy(y_pred, y_true)

    MI_XY = H_X + H_Y - H_XY
    k = len(set(y_true))
    NIT = (2 ** MI_XY) / k 
    return NIT

def entropy_modulated_accuracy(y_true, y_pred):
    """
    Calculate Entropy Modulated Accuracy (EMA) based on true and predicted labels.
    EMA is defined in Valverde-Albacete and Peláez-Moreno, 2014

    Parameters:
    - y_true : 1d array-like
      Ground truth (correct) target values.
    - y_pred : 1d array-like
      Estimated targets as returned by a classifier.
    
    Returns:
    - EMA: float, Entropy Modulated Accuracy
    """
    H_XgivenY = conditional_shannon_entropy(y_true, y_pred)
    ema = 2 ** -H_XgivenY
    return ema
