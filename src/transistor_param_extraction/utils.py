"""
Utility functions for neural network operations.

This module contains helper functions for data manipulation, metrics calculation,
and other utility operations.
"""

import numpy as np
from sklearn.utils import shuffle


def calc_R2(y_true, y_pred):
    """
    Calculate the coefficient of determination R^2 for real vs. predicted data.

    Parameters:
        y_true (array-like object)
            --  Our ground truth data.
        y_pred (array-like object):
            --  Our predicted data.

    Returns:
        The R2 between y_true and y_pred as a float.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


def shuffle_arrays_in_unison(Xarr, Yarr):
    """
    Takes two arrays, shuffles them together (preserving their relative order)
    and returns the shuffled arrays.

    Parameters:
        Xarr (numpy array): First array to shuffle
        Yarr (numpy array): Second array to shuffle

    Returns:
        Xarr, Yarr (tuple): Shuffled arrays in unison
    """
    indices = np.arange(Xarr.shape[0])
    np.random.shuffle(indices)
    Xarr = Xarr[indices]
    Yarr = Yarr[indices]
    return Xarr, Yarr
