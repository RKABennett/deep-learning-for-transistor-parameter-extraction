"""
Neural network training functions for transistor parameter extraction.

This module contains functions for training both forward and inverse neural networks
with learning rate annealing and early stopping.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError

from .losses import surrogate_loss, CombinedMSELoss
from . import NN_variables as NNv

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
model_forward_pretrained = None


def train_inverse_NN(
    X_train,
    Y_train,
    X_dev,
    Y_dev,
    model_inverse,
    model_name_inverse,
    model_forward,
    lr,
    ar,
    N_anneals,
    patience,
    bs,
):
    """
    Train an inverse neural network to predict physical parameters (e.g.,
    mobility, barrier height, etc) that will allow a current simulator
    (e.g., TCAD or a compact model) to reproduce input current-voltage
    characteristics.

    Parameters:
        X_train (numpy array)
            --  3D array containing input training data, i.e., current-voltage
                curves and features. Should be formatted such that the first
                dimension corresponds to different devices, the second dimension
                corresponds to fixed Vgs points, and the third axis corresponds
                to different features.
        Y_train (numpy array)
            --  2D array containing output training data, i.e., the physical
                parameters that we wish to solve for. To evaluate our loss
                function, we require that the second and third dimensions of
                X_train be concatenated with the parameters, so each entry of
                Y_train contains many more entries than just the parameters.
                However, we discard all values except for the parameters.
        X_dev (numpy array)
            --  Same as X_train, for the development set.
        Y_dev (numpy array)
            --  Same as Y_train, for the development set
        model_inverse (tensorflow keras model)
            --  The initialized inverse model that we wish to train.
        model_name_inverse (string)
            --  The name we wish to save our model as.
        model_forward (tensorflow keras model)
            --  Pre-trained forward neural network we use to evaluate our loss
                function.
        lr (float)
            --  Initial learning rate for training.
        ar (float)
            --  Annealing rate for learning rate annealing.
        N_anneals (int)
            --  Number of annealing cycles. (Use N_anneals = 1 to avoid
                annealing.)
        patience (int)
            --  Patience used for early stopping
        bs (int)
            -- Mini-batch size

    Returns:
        The trained inverse network and the val loss history during training.
    """

    # We need the forward model to be accessible in our forward loss function
    # so here we redefine it as a global variable. This probably isn't the
    # cleanest approach, but it works for now.
    global model_forward_pretrained
    model_forward_pretrained = model_forward

    val_loss_history = []
    cp = ModelCheckpoint(dir_path + "/" + model_name_inverse, save_best_only=True)
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    # Check the val_loss before we begin training and save the starting
    # weights. If training does not improve our network, we revert back to
    # these at the end.
    model_inverse.compile(
        loss=surrogate_loss, optimizer=Adam(learning_rate=lr), jit_compile=False
    )
    starting_val_loss_original = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
    pretrained_weights_original = model_inverse.get_weights()

    # Learning rate annealing loop
    for i in range(N_anneals):

        # Check the val_loss before each training loop and save the starting
        # weights. If the loop does not improve our network, we revert back to
        # these at the end.
        starting_val_loss = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
        starting_weights = model_inverse.get_weights()

        # Setting jit_compile=False is necessary to avoid an error when using
        # a GRU-based forward neural network to evaluate our loss function
        # for a dense inverse NN. This could be system dependent; if you
        # encounter a compilation error during training, removing jit_compile
        # below could be a good starting point.
        model_inverse.compile(
            loss=surrogate_loss, optimizer=Adam(learning_rate=lr), jit_compile=False
        )

        # We set a huge number of epochs because we train with early stopping.
        model_fit = model_inverse.fit(
            X_train,
            Y_train,
            validation_data=(X_dev, Y_dev),
            epochs=10**10,
            callbacks=[cp, es],
            batch_size=bs,
        )

        val_loss_history.extend(model_fit.history["val_loss"])
        lr *= ar

        # Check the val loss after the training loop and compare it to before
        # the training loop. If the val loss has gotten worse, we return to
        # the weights before the training loop.
        current_val_loss = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
        current_weights = model_inverse.get_weights()
        print(
            "Starting val loss = {}, current val_loss = {}".format(
                starting_val_loss, np.min(val_loss_history)
            )
        )

        if starting_val_loss < current_val_loss:
            print("Resetting weights for this cycle")
            model_inverse.set_weights(starting_weights)
        else:
            print("Updating weights for this cycle")
            model_inverse.load_weights(dir_path + "/" + model_name_inverse)

    # Display helpful information after training is complete.
    print("TRAINING COMPLETE.")

    print(
        "Pretrain val loss = {}, current val_loss = {}".format(
            starting_val_loss, np.min(val_loss_history)
        )
    )

    # Reset our weights if the full training cycle did not improve our val loss
    if starting_val_loss_original < np.min(val_loss_history):
        print("Resetting weights")
        model_inverse.set_weights(pretrained_weights_original)
    else:
        print("Updating weights")
        model_inverse.load_weights(dir_path + "/" + model_name_inverse)

    return model_inverse, val_loss_history


def train_forward_NN(
    Id_train,
    params_train,
    Id_dev,
    params_dev,
    model_forward,
    model_name_forward,
    lr,
    ar,
    N_anneals,
    patience,
    bs,
):
    """
    Train a forward neural network to predict current-voltage characteristics
    based on input parameters such as mobility and Schottky barrier height.
    This network mimics a physics-based TCAD model or a compact model; we use
    it to generate a pre-training set and to evaluate the loss function of
    our inverse neural network.

    Parameters:
        Id_train (numpy array)
            --  3D array containing OUTPUT training data, i.e., current-voltage
                characteristics. The first dimension corresponds to the device;
                the second to the fixed Vgs grid, and the third to the current
                itself. Here, along that third axis, we consider the linear-
                and log10 of Id at each Vds considered, giving a total of
                2*(number of Vds measured) features.

        params_train (numpy array)
            --  2D array containing the INPUT training data, i.e., the physical
                parameters that our current-voltage model accepts, e.g.,
                mobility, Schottky barrier height.
        Id_dev (numpy array)
            --  Same as Id_train, for the development set.
        params_dev (numpy array)
            --  Same as params_train, for the development set
        model_forward (tensorflow keras model)
            --  The initialized forward model that we wish to train.
        model_name_forward (string)
            --  The name we wish to save our model as.
        lr (float)
            --  Initial learning rate for training.
        ar (float)
            --  Annealing rate for learning rate annealing.
        N_anneals (int)
            --  Number of annealing cycles. (Use N_anneals = 1 to avoid
                annealing.)
        patience (int)
            --  Patience used for early stopping
        bs (int)
            -- Mini-batch size

    Returns:
        The trained forward network and the val loss history during training.
    """

    val_loss_history = []
    cp = ModelCheckpoint(dir_path + "/" + model_name_forward, save_best_only=True)
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    model_forward.compile(
        loss=CombinedMSELoss(),
        optimizer=Adam(learning_rate=lr),
        metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()],
    )

    # Learning rate annealing loop
    for i in range(N_anneals):
        # Check the val_loss before each training loop and save the starting
        # weights. If the loop does not improve our network, we revert back to
        # these at the end.
        starting_val_loss = model_forward.evaluate(params_dev, Id_dev, verbose=0)[0]
        starting_weights = model_forward.get_weights()

        model_forward.compile(
            loss=CombinedMSELoss(),
            optimizer=Adam(learning_rate=lr),
            metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()],
        )

        # We set a huge number of epochs because we train with early stopping.
        model_fit = model_forward.fit(
            params_train,
            Id_train,
            validation_data=(params_dev, Id_dev),
            epochs=10**10,
            callbacks=[cp, es],
            batch_size=bs,
        )

        val_loss_fn = np.min(model_fit.history["val_loss"])
        val_loss_history.append(val_loss_fn)
        lr = lr * ar

        # Check the val loss after the training loop and compare it to before
        # the training loop. If the val loss has gotten worse, we return to
        # the weights before the training loop.
        current_val_loss = model_forward.evaluate(params_dev, Id_dev, verbose=0)[0]
        current_weights = model_forward.get_weights()
        print(
            "Start val loss = {}, current val_loss = {}".format(
                starting_val_loss, np.min(val_loss_history)
            )
        )

        if starting_val_loss < current_val_loss:
            print("Resetting weights for this cycle")
            model_forward.set_weights(starting_weights)
        else:
            print("Updating weights for this cycle")
            model_forward.load_weights(dir_path + "/" + model_name_forward)

    return model_forward, val_loss_history
