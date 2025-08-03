"""
Data scaling and unscaling functions for neural network training.

This module provides min-max scaling functions for arrays and vectors
to normalize data between -1 and 1.
"""

import copy
import numpy as np


def scale_X(arr, minarrs=False, maxarrs=False):
    """
    Min-max scale an array from -1 to +1. Each feature is scaled independently.

    Parameters:
        arr (numpy array)
            --  3D Array that we wish to scale. Features of arr correspond to
                its last axis.
        minarrs and maxarrs (boolean or list)
            --  If minarrs is set to False, then we calculate the minimum and
                maximum of each feature and scale between -1 and 1. If minarrs
                and maxarrs are given as lists, we use them to min-max scale
                each feature of the array following:

                arr[:,:,i]_scaled =
                    [2*(arr[:,:,i] - min(arr[:,:,i]) /
                    (max(arr[:,:,i] - min(arr[:,:,i])))] - 1
    Returns:
        scaled_arr (numpy array)
            --  The feature-wise scaled array.
        minarrs and maxarrays (list)
            --  Lists of the minimum and maximum values of the arrays BEFORE
                we scale them.
    """
    num_feats = np.shape(arr)[-1]
    scaled_arr = copy.deepcopy(arr)

    if type(minarrs) == type(False):
        minarrs = []
        maxarrs = []
        for i in range(num_feats):
            minarrs.append(np.min(arr[:, :, i]))
            maxarrs.append(np.max(arr[:, :, i]))

    for i in range(num_feats):
        scaled_arr[:, :, i], _, _ = scale_vector(arr[:, :, i], minarrs[i], maxarrs[i])
    return scaled_arr, minarrs, maxarrs


def scale_Y(arr, minarrs=False, maxarrs=False):
    """
    Min-max scale 2D array from -1 to +1. Each feature is scaled independently.

    Parameters:
        arr (numpy array)
            --  2D Array that we wish to scale. Features of arr correspond to
                its last axis.
        minarrs and maxarrs (boolean or list)
            --  If minarrs is set to False, then we calculate the minimum and
                maximum of each feature and scale between -1 and 1. If minarrs
                and maxarrs are given as lists, we use them to min-max scale
                each feature of the array following:

                arr[:,i]_scaled =
                    [2*(arr[:,i] - min(arr[:,i]) /
                    (max(arr[:,i] - min(arr[:,i])))] - 1
    Returns:
        scaled_arr (numpy array)
            --  The feature-wise scaled array.
        minarrs and maxarrays (list)
            --  Lists of the minimum and maximum values of the arrays BEFORE
                we scale them.
    """
    num_feats = np.shape(arr)[-1]
    scaled_arr = copy.deepcopy(arr)

    if not minarrs:
        minarrs = []
        maxarrs = []
        for i in range(num_feats):
            minarrs.append(np.min(arr[:, i]))
            maxarrs.append(np.max(arr[:, i]))

    for i in range(num_feats):
        scaled_arr[:, i], _, _ = scale_vector(arr[:, i], minarrs[i], maxarrs[i])
    return scaled_arr, minarrs, maxarrs


def scale_vector(arr, minarr, maxarr):
    """
    Min-max scale a vector from -1 to +1.

    Parameters:
        arr (numpy array)
            --  1D vector that we wish to scale.
        minarr, maxarr (floats)
            --  Min and max values to scale according to

    Returns:
        scaled_arr (numpy array)
            --  The scaled vector.
        minarr, maxarr (floats)
            --  The minimum and maximum values used for scaling
    """
    scaled_X = (arr - minarr) / (maxarr - minarr)
    scaled_X = 2 * scaled_X - 1
    return scaled_X, minarr, maxarr


def unscale_vector(arr, minarr, maxarr):
    """
    Unscale a vector that has previously been min-max scaled.

    Parameters:
        arr (numpy array)
            --  1D vector that we wish to unscale.
        minarr, maxarr (floats)
            --  The maximum and minimum that we wish to unscale according to.
    Returns:
        unscaled_arr (numpy array)
            --  The unscaled array
    """
    arr = (arr + 1) / 2
    unscaled_arr = arr * (maxarr - minarr) + minarr
    return unscaled_arr


def unscale_X(arr, minarrs, maxarrs):
    """
    Unscale an array that has previously been min-max scaled.

    Parameters:
        arr (numpy array)
            --  2D array that we wish to unscale. Here, features correspond to
                the last index of arr.
        minarrs and maxarrs (lists)
            --  Lists of maximum and minimum values that we wish to unscale
                according to.
    Returns:
        unscaled_arr (numpy array)
            --  The unscaled array
    """
    unscaled_arr = copy.deepcopy(arr)
    for i in range(12):
        minarr = minarrs[i]
        maxarr = maxarrs[i]
        unscaled_arr[:, i] = unscale_vector(arr[:, i], minarrs[i], maxarrs[i])
    return unscaled_arr


def unscale_3D_arr(arr, minarrs, maxarrs):
    """
    Unscale an array that has previously been min-max scaled.

    Parameters:
        arr (numpy array)
            --  3D array that we wish to unscale. Here, features correspond to
                the last index of arr.
        minarrs and maxarrs (lists)
            --  Lists of  maximum and minimum values that we wish to unscale
                according to.
    Returns:
        unscaled_arr (numpy array)
            --  The unscaled array
    """
    unscaled_arr = copy.deepcopy(arr)
    for i in range(12):
        minarr = minarrs[i]
        maxarr = maxarrs[i]
        unscaled_arr[:, :, i] = unscale_vector(arr[:, :, i], minarrs[i], maxarrs[i])
    return unscaled_arr


def unscale_predicted(predicted_current, xmins, xmaxs):
    """
    Unscale predicted current values.

    Parameters:
        predicted_current (array): Scaled predicted current
        xmins, xmaxs (arrays): Min/max scaling parameters

    Returns:
        unscaled (array): Unscaled current predictions
    """
    unscaled = []
    for i in range(4):
        unscaled_vec = unscale_vector(predicted_current[:, i], xmins[i], xmaxs[i])
        unscaled.append(unscaled_vec)
    return np.array(unscaled)
