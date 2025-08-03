"""
Custom loss functions for neural network training.

This module contains specialized loss functions for forward and inverse neural networks
used in transistor parameter extraction.
"""

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from . import NN_variables as NNv


def surrogate_loss(Y_true, Y_pred):
    """
    Custom loss function for our inverse neural network. Our loss here is:

        sqrt(MSE*L_Id)

    where

        MSE is the standard mean square error for our model parameters.

        L_Id is a term that describes how much error we have in our
        original and predicted current based on the parameters that we are
        estimating.

    Here, we evaluate L_Id by first computing the true and predicted currents:

        Id_true = measured current
        Id_predicted = f(Y_predicted)
        where f is a pre-trained forward neural network; it must be a globally
        defined variable named 'model forward pretrained' so that we can
        access it here.

    and then L_Id is calculated based on the difference of the actual vs.
    predicted current, and its 1st and 2nd derivatives, in both linear and
    log space.

    Note that Id_true is our input to the inverse neural network. There is no
    direct way to access the network's inputs while evaluating the loss fn;
    thus, before training begins, we concatenate the MOSFET physical parameters
    with the current into a combined vector, which we feed into the NN as our
    intended output vector. We use the values of Id in the output vector only
    when evaluating Lid and discard them for the rest of the loss fn.
    """

    if "model_forward_pretrained" not in globals():
        raise NameError(
            """The forward model needs to be a global variable
                         named \'model_forward_pretrained\' """
        )

    # take only the few relevant parameters at the start of the Y vectors to
    # evaluate the MSE of the parameter error and the current values to
    # evaluate L_id
    mse_Y = MeanSquaredError()(
        Y_true[:, 0 : NNv.num_params], Y_pred[:, 0 : NNv.num_params]
    )
    Id_true = Y_true[:, 8:]

    Id_pred = model_forward_pretrained(Y_pred[:, 0 : NNv.num_params])
    Id_pred = tf.transpose(Id_pred, perm=[0, 2, 1])
    Id_pred = tf.reshape(Id_pred, [-1, 2 * NNv.num_IdVg * NNv.n_points])
    mse_Id = MeanSquaredError()(Id_true, Id_pred)

    Id_true_1st_deriv = Id_true[:, 1:] - Id_true[:, :-1]
    Id_pred_1st_deriv = Id_pred[:, 1:] - Id_pred[:, :-1]
    mse_1st_deriv = MeanSquaredError()(Id_true_1st_deriv, Id_pred_1st_deriv)

    Id_true_2nd_deriv = Id_true_1st_deriv[:, 1:] - Id_true_1st_deriv[:, :-1]
    Id_pred_2nd_deriv = Id_pred_1st_deriv[:, 1:] - Id_pred_1st_deriv[:, :-1]
    mse_2nd_deriv = MeanSquaredError()(Id_true_2nd_deriv, Id_pred_2nd_deriv)

    total_loss = mse_Y**0.5 * (mse_Id + mse_1st_deriv + mse_2nd_deriv) ** 0.5

    return total_loss


class CombinedMSELoss(tf.keras.losses.Loss):
    """
    Custom loss function for our forward neural network. Our loss here is:

        MSE(Id) + MSE(delta (Id)) + MSE(delta (delta (Id)))

    i.e., similar to MSE(Id) + MSEs of its first and second derivatives.
    """

    def call(self, Y_true, Y_pred):
        mse_loss = MeanSquaredError()

        mse_Id = mse_loss(Y_true, Y_pred)

        delta_Y_true = Y_true[:, 1:] - Y_true[:, :-1]
        delta_Y_pred = Y_pred[:, 1:] - Y_pred[:, :-1]
        mse_deltaId = mse_loss(delta_Y_true, delta_Y_pred)

        deltadelta_Y_true = delta_Y_true[:, 1:] - delta_Y_true[:, :-1]
        deltadelta_Y_pred = delta_Y_pred[:, 1:] - delta_Y_pred[:, :-1]
        mse_deltadeltaId = mse_loss(deltadelta_Y_true, deltadelta_Y_pred)

        total_loss = mse_Id + mse_deltaId + mse_deltadeltaId
        return total_loss
