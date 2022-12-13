# coding=utf-8
import numpy as np


# %%

def logistic_loss(w, X, y, return_arr=None):
    """
    This function is used from scikit-learn source code below.

    ----------
    Parameters

    param w: ndarray, shape (n_features + 1,). Coefficient vector of the logistic function.
    param X: {array-like, sparse matrix}, shape (n_samples, n_features). Training data.
    param y:  ndarray, shape (n_samples,) Array of labels.
    return: Logistic loss, that is the negative of the log of the logistic function.

    ----------
    Source code at:

    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    """
    if np.isnan(np.sum(w)):
        if return_arr:
            return np.ones_like(y) * np.NAN
        else:
            return np.NAN

    wx = np.dot(X, w)
    yz = y * wx

    if return_arr:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out


def log_logistic(X):
    """
    This function is used from scikit-learn source code. Source link below.

    ----------
    Parameters

    param X: array-like, shape (M, N). Argument to the logistic function.
    return: array, shape (M, N). Log of the logistic function, ``log(1 / (1 + e ** -x))``
    evaluated at every point in x.

    ----------
    Notes

    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    ----------
    Source code at:

    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    """
    if X.ndim > 1:
        raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data type

    idx = X > 0

    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out
