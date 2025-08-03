"""
Data augmentation functions for neural network training.

This module provides functions to generate augmented training data using
pre-trained forward neural networks.
"""

import copy
import numpy as np
from . import NN_variables as NNv
from .scaling import scale_X, unscale_vector
import os
import sys

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))


def augment_data(
    model_forward, N_augment, N_features, Xscaling_name, Yscaling_name, V, save=True
):
    """
    Generate augmented training data using our forward neural network.

    Parameters:
        model_forward (tensorflow keras model)
            --  The trained forward neural network used to generate data.
        N_augment (int)
            --  The number of devices in the desired dataset.
        N_features (int)
            --  Number of parameter features
        Xscaling_name (str)
            --  Filepath + name for the X array (current-voltage) scaling
                parameters used when generating the original dataset.
        Yscaling_name (str)
            --  Same as above, for the Y array.
        V (array-like)
            --  Array or list corresponding to the fixed Vgs points we used.
        save (bool)
            --  Whether to save the augmented data

    Returns:
        X, Y (tuple): Augmented data in X and Y arrays.
    """
    # generate random input parameters and then call forward model to predict
    # the current-voltage characteristics
    Y = np.random.uniform(-1, 1, (N_augment, N_features))
    print(np.shape(Y))
    currents_generated = np.array(model_forward.predict(Y))

    # load scaling parameters
    Xscaling = np.loadtxt(dir_path + "/" + Xscaling_name)
    Xmins = Xscaling[0, :]
    Xmaxs = Xscaling[1, :]
    Yscaling = np.loadtxt(dir_path + "/" + Yscaling_name)
    Ymins = Yscaling[0, :]
    Ymaxs = Yscaling[1, :]

    # unscale the currents
    currents_unscaled = copy.deepcopy(currents_generated)
    for i in range(NNv.num_IdVg * 2):
        Xmin = Xmins[i]
        Xmax = Xmaxs[i]
        currents_unscaled[:, :, i] = unscale_vector(
            currents_generated[:, :, i], Xmin, Xmax
        )

    # build a new X array from the unscaled currents
    X = []
    Id = currents_unscaled[:, :, ::2]
    Id_log = currents_unscaled[:, :, 1::2]

    Id_grad = np.gradient(Id, V, axis=1, edge_order=2)
    Id_log_grad = np.gradient(Id_log, V, axis=1, edge_order=2)

    X_array = np.empty((N_augment, NNv.num_IdVg * 4, NNv.n_points))
    for j in range(NNv.num_IdVg):
        X_array[:, j * 4, :] = Id[:, :, j]
        X_array[:, j * 4 + 1, :] = Id_log[:, :, j]
        X_array[:, j * 4 + 2, :] = Id_grad[:, :, j]
        X_array[:, j * 4 + 3, :] = Id_log_grad[:, :, j]

    # Work our X array into the correct formatting
    X_array = X_array.transpose(0, 2, 1)
    X = [x_input[np.newaxis, ...] for x_input in X_array]
    X = np.array(X)
    X = np.reshape(X, (N_augment, NNv.n_points, 4 * NNv.num_IdVg))

    # Right now, the X array isn't sorted properly: we want all of the currents
    # and then all of the derivatives, but it alternates between currents and
    # derivatives right now. We fix this here.
    current_indices = []
    deriv_indices = []
    for i in range(NNv.num_IdVg):
        current_indices.append(0 + i * 4)
        current_indices.append(1 + i * 4)
        deriv_indices.append(2 + i * 4)
        deriv_indices.append(3 + i * 4)
    X = np.concatenate([X[:, :, current_indices], X[:, :, deriv_indices]], axis=-1)

    # Finally, scale the new data.
    X, Xmins_new, Xmaxs_new = scale_X(
        X, minarrs=Xmins[0 : NNv.num_IdVg * 4], maxarrs=Xmaxs[0 : NNv.num_IdVg * 4]
    )

    return X, Y
