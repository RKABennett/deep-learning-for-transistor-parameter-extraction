###############################################################################
#
# Import and define functions
#
###############################################################################
import ast
import copy
import csv
import glob
import os
import random
import sys
import time
import traceback

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.interpolate import interp1d
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Attention, BatchNormalization, Bidirectional, Dense, Dropout, GRU,
    GroupNormalization, LayerNormalization, LSTM
    )
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import (MeanAbsolutePercentageError, 
                                     RootMeanSquaredError)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import NN_variables as NNv

np.random.seed(19700101)
random.seed(19700101)
tf.random.set_seed(19700101)
dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))

###############################################################################
#
# Training functions
#
###############################################################################

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
    cp = ModelCheckpoint(
                         dir_path + '/' + model_name_inverse, 
                         save_best_only=True
                         )
    es = EarlyStopping(
                       monitor='val_loss', 
                       patience=patience, 
                       restore_best_weights=True
                       )

    # Check the val_loss before we begin training and save the starting
    # weights. If training does not improve our network, we revert back to 
    # these at the end.
    model_inverse.compile(
        loss=surrogate_loss,
        optimizer=Adam(learning_rate=lr),
        jit_compile=False
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
                              loss=surrogate_loss, 
                              optimizer=Adam(learning_rate=lr), 
                              jit_compile=False 
                              )

        # We set a huge number of epochs because we train with early stopping.
        model_fit = model_inverse.fit(
                                      X_train, 
                                      Y_train, 
                                      validation_data=(X_dev, Y_dev),
                                      epochs=10**10,
                                      callbacks=[cp, es], 
                                      batch_size=bs
                                      )

        val_loss_history.extend(model_fit.history['val_loss'])
        lr *= ar


        # Check the val loss after the training loop and compare it to before
        # the training loop. If the val loss has gotten worse, we return to
        # the weights before the training loop.
        current_val_loss = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
        current_weights = model_inverse.get_weights()
        print("Starting val loss = {}, current val_loss = {}".format(
                                                        starting_val_loss, 
                                                        np.min(val_loss_history)
                                                        )
                                                        )
        
        if starting_val_loss < current_val_loss:
            print("Resetting weights for this cycle")
            model_inverse.set_weights(starting_weights)
        else:
            print("Updating weights for this cycle")
            model_inverse.load_weights(dir_path + '/' + model_name_inverse)
    
    # Display helpful information after training is complete.
    print("TRAINING COMPLETE.".format(
                                      starting_val_loss, 
                                      np.min(val_loss_history)
                                      )
                                      )

    print("Pretrain val loss = {}, current val_loss = {}".format(
                                                       starting_val_loss, 
                                                       np.min(val_loss_history)
                                                       )
                                                       )

    # Reset our weights if the full training cycle did not improve our val loss
    if starting_val_loss_original < np.min(val_loss_history):
        print("Resetting weights")
        model_inverse.set_weights(pretrained_weights_original)
    else:
        print("Updating weights")
        model_inverse.load_weights(dir_path + '/' + model_name_inverse)
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
                     bs
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
        The trained forward network and the val loss history during training.
    """

    val_loss_history = []
    cp = ModelCheckpoint(
                         dir_path + '/' + model_name_forward,
                         save_best_only=True
                         )
    es = EarlyStopping(
                       monitor='val_loss',
                       patience=patience,
                       restore_best_weights=True
                       )

    model_forward.compile(
                  loss=CombinedMSELoss(),
                  optimizer=Adam(learning_rate=lr),
                  metrics = [RootMeanSquaredError(),
                             MeanAbsolutePercentageError()]
                  )

    # Learning rate annealing loop
    for i in range(N_anneals):
        # Check the val_loss before each training loop and save the starting
        # weights. If the loop does not improve our network, we revert back to 
        # these at the end.
        starting_val_loss = model_forward.evaluate(
                                                   params_dev, 
                                                   Id_dev, 
                                                   verbose=0)[0]
        starting_weights = model_forward.get_weights()

        model_forward.compile(
                      loss=CombinedMSELoss(),
                      optimizer=Adam(learning_rate=lr),
                      metrics = [RootMeanSquaredError(),
                                 MeanAbsolutePercentageError()]
                      )
        # We set a huge number of epochs because we train with early stopping.
        model_fit = model_forward.fit(
                              params_train,
                              Id_train,
                              validation_data=(params_dev,Id_dev),
                              epochs=10**10,
                              callbacks=[cp,es],
                              batch_size = bs
                              )
        val_loss_fn = np.min(model_fit.history['val_loss'])
        val_loss_history.append(val_loss_fn)
        lr = lr*ar

        # Check the val loss after the training loop and compare it to before
        # the training loop. If the val loss has gotten worse, we return to
        # the weights before the training loop.
        current_val_loss = model_forward.evaluate(
                                                  params_dev, 
                                                  Id_dev, 
                                                  verbose=0
                                                  )[0]
        current_weights = model_forward.get_weights()
        print("Start val loss = {}, current val_loss = {}".format(
                                                        starting_val_loss, 
                                                        np.min(val_loss_history)
                                                        )
                                                        )

        if starting_val_loss < current_val_loss:
            print("Resetting weights for this cycle")
            model_forward.set_weights(starting_weights)
        else:
            print("Updating weights for this cycle")
            model_forward.load_weights(dir_path + '/' + model_name_forward)

    return model_forward, val_loss_history

###############################################################################
#
# Custom loss functions
#
###############################################################################

def surrogate_loss(Y_true, Y_pred):
    '''
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
    '''

    if 'model_forward_pretrained' not in globals():
        raise NameError('''The forward model needs to be a global variable
                         named \'model_forward_pretrained\' ''')

    # take only the few relevant parameters at the start of the Y vectors to
    # evaluate the MSE of the parameter error and the current values to
    # evaluate L_id
    mse_Y = MeanSquaredError()(
                               Y_true[:, 0:NNv.num_params], 
                               Y_pred[:, 0:NNv.num_params]
                               ) 
    Id_true = Y_true[:, 8:] 

    Id_pred = model_forward_pretrained(Y_pred[:, 0:NNv.num_params])
    Id_pred = tf.transpose(Id_pred, perm=[0, 2, 1])
    Id_pred = tf.reshape(Id_pred, [-1, 2*NNv.num_IdVg*NNv.n_points])    
    mse_Id = MeanSquaredError()(Id_true, Id_pred)

    Id_true_1st_deriv = Id_true[:, 1:] - Id_true[:, :-1]
    Id_pred_1st_deriv = Id_pred[:, 1:] - Id_pred[:, :-1]
    mse_1st_deriv = MeanSquaredError()(Id_true_1st_deriv, Id_pred_1st_deriv)

    Id_true_2nd_deriv = Id_true_1st_deriv[:, 1:] - Id_true_1st_deriv[:, :-1]
    Id_pred_2nd_deriv = Id_pred_1st_deriv[:, 1:] - Id_pred_1st_deriv[:, :-1]
    mse_2nd_deriv = MeanSquaredError()(Id_true_2nd_deriv, Id_pred_2nd_deriv)

    total_loss = mse_Y**0.5*(mse_Id + mse_1st_deriv + mse_2nd_deriv)**0.5

    return(total_loss)


class CombinedMSELoss(tf.keras.losses.Loss):

    '''
    Custom loss function for our inverse neural network. Our loss here is:

        MSE(Id) + MSE(delta (Id)) + MSE(delta (delta (Id)))
    
    i.e., similar to MSE(Id) + MSEs of its first and second derivatives.

    '''
    def call(self, Y_true, Y_pred):
        mse_loss = MeanSquaredError()

        mse_Id = mse_loss(Y_true, Y_pred)

        delta_Y_true = (Y_true[:, 1:] - Y_true[:, :-1]) 
        delta_Y_pred = (Y_pred[:, 1:] - Y_pred[:, :-1]) 
        mse_deltaId = mse_loss(delta_Y_true, delta_Y_pred)
        
        deltadelta_Y_true = (delta_Y_true[:, 1:] - delta_Y_true[:, :-1])
        deltadelta_Y_pred = (delta_Y_pred[:, 1:] - delta_Y_pred[:, :-1])
        mse_deltadeltaId = mse_loss(deltadelta_Y_true, deltadelta_Y_pred)

        total_loss = mse_Id + mse_deltaId + mse_deltadeltaId
        return total_loss

###############################################################################
#
# Data augmenting
#
###############################################################################

def augment_data(
                 model_forward, 
                 N_augment, 
                 N_features,
                 Xscaling_name, 
                 Yscaling_name,
                 V,
                 save = True
                 ):
    
    """
    Generate augmented training data using our forward neural network.

    Parameters:
        model_forward (tensorflow keras model)
            --  The trained forward neural network used to generate data.
        N_augment (int)
            --  The number of devices in the desired dataset.
        Xscaling_name (str)
            --  Filepath + name for the X array (current-voltage) scaling 
                parameters used when generating the original dataset.
        Xscaling_name (str)
            --  Same as above, for the Y array.
        V (array-like)
            --  Array or list corresponding to the fixed Vgs points we used.
    
    Returns:
        Augmented data in X and Y arrays.
    """
    # generate random input parameters and then call forward model to predict
    # the current-voltage characteristics
    Y = np.random.uniform(-1, 1, (N_augment, N_features))
    print(np.shape(Y))
    currents_generated = np.array(model_forward.predict(Y))

    # load scaling parameters
    Xscaling = np.loadtxt(dir_path + '/' + Xscaling_name)
    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]
    Yscaling = np.loadtxt(dir_path + '/' + Yscaling_name)
    Ymins = Yscaling[0,:]
    Ymaxs = Yscaling[1,:]

    # unscale the currents
    currents_unscaled = copy.deepcopy(currents_generated)
    for i in range(NNv.num_IdVg*2):
        Xmin = Xmins[i]
        Xmax = Xmaxs[i]
        currents_unscaled[:,:,i] = unscale_vector(
                                                  currents_generated[:,:,i], 
                                                  Xmin, 
                                                  Xmax
                                                  )

    # build a new X array from the unscaled currents
    X = []
    Id     = currents_unscaled[:, :, ::2]
    Id_log = currents_unscaled[:, :, 1::2]

    Id_grad     = np.gradient(Id, V, axis=1, edge_order=2)
    Id_log_grad = np.gradient(Id_log, V, axis=1, edge_order=2)

    X_array = np.empty((N_augment, NNv.num_IdVg * 4, NNv.n_points))
    for j in range(NNv.num_IdVg):
        X_array[:, j*4,     :] = Id[:, :, j]
        X_array[:, j*4 + 1, :] = Id_log[:, :, j]
        X_array[:, j*4 + 2, :] = Id_grad[:, :, j]
        X_array[:, j*4 + 3, :] = Id_log_grad[:, :, j]

    # Work our X array into the correct formatting
    X_array = X_array.transpose(0, 2, 1) 
    X = [x_input[np.newaxis, ...] for x_input in X_array]
    X = np.array(X)
    X = np.reshape(X, (N_augment, NNv.n_points, 4*NNv.num_IdVg))
    
    # Right now, the X array isn't sorted properly: we want all of the currents
    # and then all of the derivatives, but it alternates between currents and
    # derivatives right now. We fix this here.
    current_indices = []
    deriv_indices = []
    for i in range(NNv.num_IdVg):
        current_indices.append(0+i*4) 
        current_indices.append(1+i*4) 
        deriv_indices.append(2+i*4)
        deriv_indices.append(3+i*4) 
    X = np.concatenate([
                     X[:,:, current_indices],
                     X[:,:, deriv_indices]
                     ],
                     axis=-1) 

    # Finally, scale the new data.
    X, Xmins_new, Xmaxs_new = scale_X(
                                      X, 
                                      minarrs = Xmins[0:NNv.num_IdVg*4], 
                                      maxarrs = Xmaxs[0:NNv.num_IdVg*4]
                                      )
    
    return X, Y


###############################################################################
#
# Functions for testing models
#
###############################################################################

def test_model_inverse_params(
    X_test_dummy,
    Y_test_dummy,
    model_inverse,
    model_name_inverse,
    Xscaling_name,
    Yscaling_name,
    ):

    '''
    Test the inverse model by applying the trained inverse NN to test data and
    extracting the actual vs. predicted params.
    '''
   
    X_test = copy.deepcopy(X_test_dummy)
    Y_test = copy.deepcopy(Y_test_dummy)

    Xscaling = np.loadtxt(dir_path + '/' + Xscaling_name)
    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]

    Yscaling = np.loadtxt(dir_path + '/' + Yscaling_name)
    Ymin = Yscaling[0,:]
    Ymax = Yscaling[1,:]

    Y_pred = np.array(model_inverse.predict(X_test))[:, 0:NNv.num_params]
    errors = []

    for i in range(np.shape(Y_test)[1]):
        Y_test[:,i] = unscale_vector(Y_test[:,i], Ymin[i], Ymax[i])        
        Y_pred[:,i] = unscale_vector(Y_pred[:,i], Ymin[i], Ymax[i])        


        error = (Y_pred[:,i] - Y_test[:,i])
        if i in [2,6]:
            error /= 1e19
        elif i in [3]:
            error /= 1e13

        errors.append(error)
        std_error = np.std(error)
        abs_error = np.abs(error)
        
        fig,ax = plt.subplots(1,2)
        
        # actual vs. predicted
        ax[0].plot(
                    Y_test[:,i], 
                    Y_pred[:,i],
                    color='k', 
                    marker='o', 
                    markersize = 0.25,
                    ls='None'
                    )

        # line with slope = 1 for reference 
        ax[0].plot(
                       [Ymin[i], Ymax[i]], 
                       [Ymin[i], Ymax[i]], 
                       color='r', 
                       marker='None', 
                       ls='--',
                       linewidth = 1.0
                       ) 

        range_hist = 5*np.std(error)
        
        # histogram of errors    
        ax[1].hist(
                 error, 
                 bins = np.linspace(-range_hist, range_hist, 20),
                 density = False,
                 edgecolor = 'k', 
                 linewidth=0.25,
                 color = '#1048a2'
                 )

        ax[0].set_xlabel('Actual quantity')
        ax[0].set_ylabel('Predicted quantity')


        ax[1].set_xlim([-range_hist, range_hist])
        ax[1].set_xlabel('Percent error')
        ax[1].set_ylabel('Count')


        fig.suptitle('''
                     Median abs error = {:.4f} \n
                     Mean abs error = {:.4f} \n
                     Stdev of error = {:.4f} \n
                     '''.format(
                                np.median(abs_error), 
                                np.mean(abs_error), 
                                np.std(error))
                     )


        plt.tight_layout()
        plt.savefig(dir_path + '/{}_error_plot_idx={}.png'.format(model_name_inverse, i))
        plt.close()

    return errors
    
    # this section can be added back in to inspect defect profiles and carrier
    # densities
    for i in range(10):
        print(i)
        DOS_actual = Y_test[i,2] * 0.615e-7 # units of cm^-2
        Dpeak_actual = Y_test[i,3]
        Dmid_actual = Y_test[i,4]
        Dstd_actual = Y_test[i,5]
        Apeak_actual = Y_test[i,6] * 0.615e-7 # units of cm^-2
        Astd_actual = Y_test[i,7]

        DOS_pred = Y_pred[i,2] * 0.615e-7 # units of cm^-2
        Dpeak_pred = Y_pred[i,3]
        Dmid_pred = Y_pred[i,4]
        Dstd_pred = Y_pred[i,5]
        Apeak_pred = Y_pred[i,6] * 0.615e-7 # units of cm^-2
        Astd_pred = Y_pred[i,7]
        
        plot_name = 'profile_{}.png'.format(i)
        E = np.linspace(-2, 2, 1000)

        donors_actual, acceptors_actual = calc_profile(E, 
                                                      Dpeak_actual, 
                                                      Dmid_actual, 
                                                      Dstd_actual, 
                                                      Apeak_actual, 
                                                      Astd_actual
                                                      )



        donors_pred, acceptors_pred = calc_profile(E, 
                                                      Dpeak_pred, 
                                                      Dmid_pred, 
                                                      Dstd_pred, 
                                                      Apeak_pred, 
                                                      Astd_pred
                                                      )
        


        mask = np.where(E <= 0)
        DOS_actual_profile = DOS_actual*np.ones(np.size(E))
        DOS_actual_profile[mask] = 0
        DOS_pred_profile = DOS_pred*np.ones(np.size(E))
        DOS_pred_profile[mask] = 0
        plot_profile(
                     E,
                     DOS_actual_profile,
                     DOS_pred_profile,
                     donors_actual,
                     donors_pred,
                     acceptors_actual,
                     acceptors_pred,
                     plot_name)


        plot_name = 'nch_{}.png'.format(i)

        nch_actual = calc_nch_vs_Ef(E, E, DOS_actual_profile, donors_actual, acceptors_actual)
        nch_pred = calc_nch_vs_Ef(E, E, DOS_pred_profile, donors_pred, acceptors_pred)
        
        plt.semilogy(
                E,
                nch_actual,
                color = 'k',
                ls = '-',
                marker = 'None'
                )
        
        plt.semilogy(
                E,
                nch_pred,
                color = 'r',
                ls = '--',
                marker = 'None'
                )

        plt.xlim(-1, 0.5)
        plt.savefig(dir_path + '/{}'.format(plot_name))
        plt.close()

def test_model_inverse_current(
    X_test,
    Y_test,
    model_forward,
    model_inverse,
    Xscaling_name,
    Yscaling_name,
    error_metric,
    error_filename,
    deriv_error = True,
    plot = True,
    plot_range = range(1),
    save_fits = False,
    fit_name = False 
    ):

    '''
    Test the inverse model by applying the trained inverse NN to the test data
    and extracting out the predicted parameters, Y_pred. After, we take a 
    pre-trained forward model, f, and test the accuracy of Id by comparing 
    f(Y_true) and f(Y_pred). This approach assumes we are confident in 
    our forward model. 

    Parameters:
        X_test (numpy array) 
            --  3D array containing input test data, i.e., current-voltage
                characteristics. The first dimension corresponds to the device;
                the second to the fixed Vgs grid, and the third to the current
                itself. Here, along that third axis, we consider the linear-
                and log10 of Id at each Vds considered, giving a total of
                2*(number of Vds measured) features.
        Y_test (numpy array)
            --  2D array containing the output testing data, i.e., the physical
                parameters that our current-voltage model accepts, e.g., 
                mobility, Schottky barrier height.
        model_forward (tensorflow keras model)
            --  The fully trained forward NN that we will use to estimate Id
                from physical parameters. 
        model_inverse (tensorflow keras model)
            --  The trained inverse NN that we will test.
        Xscaling_name (str)
            --  Filepath + name for the X array (current-voltage) scaling 
                parameters used when generating the original dataset.
        Xscaling_name (str)
            --  Same as above, for the Y array.
        error_metric (function)
            --  Function that we will use to evaluate the error in Id.
        error_filename (str)
            --  Filepath + filename that we wish to save our errors to.
        deriv_error (boolean)
            --  if True, calculate error_metric on the gradients of the true
                and predicted data
        plot (boolean)
            --  if True, save sample plots for fits.
        plot_range (range object)
            --  Range of plots we wish to save, if plot is True. e.g., use 
                range(10) to save the first 10 plots.
        save_fits (boolean)
            --  if True, save fits across the test set.
        fit_name (str)
            -- filename that the fit should be saved as, if save_fits is True
    
    Returns:
        X_test (numpy array)
            --  Values of X_test as imported, unscaled
        X_pred (numpy array)
            --  The unscaled current-voltage characteristics obtained using 
                the neural network-predicted input parameters
        errors (numpy array) 
            --  Array of errors between the true and predicted current, as 
            assessed using the provided error_metric function.

    IMPORTANT NOTE: This test calls a pre-trained forward NN to estimate Id.
    Thus, the error we extract here is only an estimate.To find the true error 
    in our Id, we need to run the Sentaurus simulation using Y_pred and 
    compare it to our original Id.
    '''
   
    Xscaling = np.loadtxt(dir_path + '/' + Xscaling_name)
    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]

    Yscaling = np.loadtxt(dir_path + '/' + Yscaling_name)
    Ymin = Yscaling[0,:]
    Ymax = Yscaling[1,:]

    Z_pred = np.array(model_inverse.predict(X_test))
    X_pred = np.array(model_forward.predict(Z_pred[:,0:NNv.num_params]))

    X_test = copy.deepcopy(X_test)
    X_pred = copy.deepcopy(X_pred)
    for i in range(4):
        Xmin = Xmins[i]
        Xmax = Xmaxs[i]
        X_test[:,:,i] = unscale_vector(X_test[:,:,i], Xmin, Xmax)
        X_pred[:,:,i] = unscale_vector(X_pred[:,:,i], Xmin, Xmax)

    errors = []
    
    

    ###########################################################################
    #
    # Extract and plot our error metric
    #
    ###########################################################################
    for i in range(np.shape(X_test)[0]):
        error = 0
        for j in range(NNv.num_IdVg*2):
            if deriv_error:
                error += error_metric(
                                      np.gradient(X_test[i,:,j]), 
                                      np.gradient(X_pred[i,:,j])
                                      )
            else:
                error += error_metric(X_test[i,:,j], X_pred[i,:,j])
        errors.append(error/(NNv.num_IdVg*2))

    np.savetxt(dir_path + '/{}'.format(error_filename), errors)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(
             errors, 
             density = False,
             edgecolor = 'k', 
             linewidth=0.25,
             color = '#1048a2'
             )


    plt.tight_layout()
    plt.savefig(dir_path + '/R2_histogram.png')
    plt.close()
    
    ###########################################################################
    #
    # Plot actual vs. predicted current
    #
    ########################################################################### 
    if plot:
        for i in plot_range:
            print('Plot number {}'.format(i))
            fig, axs = plt.subplots(2, 2, figsize=(12, 4*NNv.num_IdVg))
            for j in range(NNv.num_IdVg*2):
                row, col = divmod(j, 2)  
                axs[row, col].plot(
                                   X_test[i, :, j], 
                                   color='k', 
                                   marker='o', 
                                   ls='None'
                                   )

                axs[row, col].plot(
                                   X_pred[i, :, j], 
                                   'r', 
                                   ls='--'
                                   )

            plt.tight_layout()
            plt.savefig(dir_path + '/inverse_plot_{}.png'.format(i))
            plt.close()

    if save_fits:
        data_folder = dir_path + '/fits_inverse'
        actual_fits = dir_path + '/fits_inverse/{}_actual'.format(fit_name)
        pred_fits =   dir_path + '/fits_inverse/{}_pred'.format(fit_name)

        for dir_name in [data_folder, actual_fits, pred_fits]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        for i in range(np.shape(X_test)[0]):
            actuals = []
            preds = []
            for j in range(NNv.num_IdVg*2):
                Id_actual = X_test[i,:, j]
                Id_pred   = X_pred[i,:, j]
                actuals.append(Id_actual)
                preds.append(Id_pred)

            actual_filename = 'error={:.12f}_actual.dat'.format(errors[i])
            pred_filename = 'error={:.12f}_pred.dat'.format(errors[i])
            np.savetxt(actual_fits + '/' + actual_filename, actuals)
            np.savetxt(pred_fits + '/' + pred_filename, preds)


    return X_test, X_pred, errors

def test_model_forward(
        X_test,
        Y_test,
        model_forward,
        Xscaling_name,
        Yscaling_name,
        error_metric,
        error_filename,
        fit_name = 'forward',
        plot = True,
        plot_range = range(1)
        ):

    '''
    Test the forward model by using it to extract current-voltage curves from
    known parameters, and then comparing to the actual current measurements.

    Parameters:
        X_test (numpy array) 
            --  3D array containing OUTPUT test data, i.e., current-voltage
                characteristics. The first dimension corresponds to the device;
                the second to the fixed Vgs grid, and the third to the current
                itself. Here, along that third axis, we consider the linear-
                and log10 of Id at each Vds considered, giving a total of
                2*(number of Vds measured) features.
        Y_test (numpy array)
            --  2D array containing the INPUT testing data, i.e., the physical
                parameters that our current-voltage model accepts, e.g., 
                mobility, Schottky barrier height.
        model_forward (tensorflow keras model)
            --  The forward NN that we will test.
        Xscaling_name (str)
            --  Filepath + name for the X array (current-voltage) scaling 
                parameters used when generating the original dataset.
        Xscaling_name (str)
            --  Same as above, for the Y array.
        error_metric (function)
            --  Function that we will use to evaluate the error in Id.
        error_filename (str)
            --  Filepath + filename that we wish to save our errors to.
                and predicted data
        plot (boolean)
            --  if True, save sample plots for fits.
        plot_range (range object)
            --  Range of plots we wish to save, if plot is True. e.g., use 
                range(10) to save the first 10 plots.

    Returns:
        X_test (numpy array)
            --  Values of X_test as imported, unscaled
        X_pred (numpy array)
            --  The unscaled current-voltage characteristics output by the
                forward NN
        errors (numpy array) 
            --  Array of errors between the true and predicted current, as 
            assessed using the provided error_metric function.
    '''

    ###############################################################################
    #
    # Compile predictions and process
    #
    ###############################################################################

    X_pred = np.array(model_forward.predict(Y_test))

    Xscaling = np.loadtxt(dir_path + '/' + Xscaling_name)
    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]

    Yscaling = np.loadtxt(dir_path + '/' + Yscaling_name)
    Ymin = Yscaling[0,:]
    Ymax = Yscaling[1,:]

    counts_train = []
    counts_dev = []
    counts_test = []

    N_test = np.size(Y_test[0])
    errors = []

    X_test = copy.deepcopy(X_test)
    Y_test = copy.deepcopy(Y_test)

    for i in range(4):
        Xmin = Xmins[i]
        Xmax = Xmaxs[i]
        X_test[:,:,i] = unscale_vector(X_test[:,:,i], Xmin, Xmax)
        X_pred[:,:,i] = unscale_vector(X_pred[:,:,i], Xmin, Xmax)

    for i in range(np.shape(X_test)[0]):
        error = 0
        for j in range(np.shape(X_test)[2]):
            error += error_metric(X_test[i,:,j], X_pred[i,:,j])
        errors.append(error/np.shape(X_test)[2])

    np.savetxt(dir_path + '/{}'.format(error_filename), errors)

    ###########################################################################
    #
    # Save fits to file
    #
    ###########################################################################
    data_folder = dir_path + '/fits_forward'
    actual_fits = dir_path + '/fits_forward/{}_actual'.format(fit_name)
    pred_fits =   dir_path + '/fits_forward/{}_pred'.format(fit_name)

    for dir_name in [data_folder, actual_fits, pred_fits]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    for i in range(np.shape(X_test)[0]):
        actuals = []
        preds = []
        for j in range(4):
            Id_actual = X_test[i,:, j] 
            Id_pred   = X_pred[i,:, j]
            actuals.append(Id_actual)
            preds.append(Id_pred)

        actual_filename = 'error={:.12f}_actual.dat'.format(errors[i])
        pred_filename = 'error={:.12f}_pred.dat'.format(errors[i])
        np.savetxt(actual_fits + '/' + actual_filename, actuals)
        np.savetxt(pred_fits + '/' + pred_filename, preds)


    ###########################################################################
    #
    # Optionally plot
    #
    ###########################################################################
    if plot:
        for i in plot_range:
            print('Plot number {}'.format(i))
            fig, axs = plt.subplots(NNv.num_IdVg, 2, figsize=(12, 10))
            for j in range(np.shape(X_test)[2]):
                row, col = divmod(j, 2)  # Determine row and column for the subplot
                axs[row, col].plot(X_test[i, :, j], color='k', marker='o', ls='None')
                axs[row, col].plot(X_pred[i, :, j], 'r', ls='--')
    
            plt.tight_layout()
            plt.savefig(dir_path + '/forward_plot_{}.png'.format(i))
            plt.close()

    return X_test, X_pred, errors

def plot_forward_comparison(X_test, test_predictions, errors, plot_folder_name, V):
   
    if not os.path.exists(dir_path + '/' + plot_folder_name):
        os.makedirs(dir_path + '/' + plot_folder_name)

    for i in range(np.shape(X_test)[0]):
        start = 0
        stop = 32
        skip = 3
        R2 = errors[i]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 1.75))

        # Panel 1
        ax1a = ax1.twinx()
        ax1.plot(V[start:stop:skip], 10**6*X_test[i, :, 0][start:stop:skip], color='k', marker='o', ls='None')
        ax1.plot(V, 10**6*test_predictions[i, :, 0], 'r', ls='--')
        ax1a.semilogy(V[start:stop:skip], 10**6*np.power(10, X_test[i, :, 1])[start:stop:skip], color='k', marker='o', ls='None')
        ax1a.semilogy(V, 10**6*np.power(10, test_predictions[i, :, 1]), 'r', ls='--')

        # Panel 2
        ax2a = ax2.twinx()
        ax2.plot(V[start:stop:skip], 10**6*X_test[i, :, 2][start:stop:skip], color='k', marker='o', ls='None')
        ax2.plot(V, 10**6*test_predictions[i, :, 2], 'r', ls='--')
        ax2a.semilogy(V[start:stop:skip], 10**6*np.power(10, X_test[i, :, 3][start:stop:skip]), color='k', marker='o', ls='None')
        ax2a.semilogy(V, 10**6*np.power(10, test_predictions[i, :, 3]), 'r', ls='--')

        plt.tight_layout()

        maxId100 = np.max(X_test[i, :, 0])
        maxId1000 = np.max(X_test[i, :, 2])

        ax1.set_ylim(-0.1*10**6*maxId100, 10**6*maxId100*1.5)
        ax2.set_ylim(-0.1*10**6*maxId1000, 10**6*maxId1000*1.5)
        ax1a.set_ylim(10**-6, 10**2)
        ax2a.set_ylim(10**-6, 10**2)


        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_actual_Vds=0.1_linear.dat'.format(R2), np.array([V, X_test[i,:,0]]))
        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_pred_Vds=0.1_linear.dat'.format(R2), np.array([V, test_predictions[i,:,0]]))

        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_actual_Vds=0.1_log.dat'.format(R2), np.array([V, X_test[i,:,1]]))
        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_pred_Vds=0.1_log.dat'.format(R2), np.array([V, test_predictions[i,:,1]]))

        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_actual_Vds=1.0_linear.dat'.format(R2), np.array([V, X_test[i,:,2]]))
        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_pred_Vds=1.0_linear.dat'.format(R2), np.array([V, test_predictions[i,:,2]]))

        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_actual_Vds=1.0_log.dat'.format(R2), np.array([V, X_test[i,:,3]]))
        np.savetxt(dir_path + '/' + plot_folder_name + '/R2_{:.6f}_data_pred_Vds=1.0_log.dat'.format(R2), np.array([V, test_predictions[i,:,3]]))

        plt.tight_layout()
        plt.savefig(dir_path + '/' + plot_folder_name + '/R2_{:.6f}plot_{}.png'.format(R2,i))
        plt.close()

        print(i)


###############################################################################
#
# Misc custom functions
#
###############################################################################

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
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def shuffle_arrays_in_unison(Xarr, Yarr):
    """
    Takes two arrays, shuffles them together (preserving their relative order)
    and returns the shuffled arrays.
    """
    indices = np.arange(Xarr.shape[0])
    np.random.shuffle(indices)
    Xarr = Xarr[indices]
    Yarr = Yarr[indices]
    return Xarr, Yarr


###############################################################################
#
# Scaling functions
#
###############################################################################

def scale_X(arr, minarrs=False, maxarrs=False):
    '''
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
    '''
    num_feats = np.shape(arr)[-1]
    scaled_arr = copy.deepcopy(arr)

    if type(minarrs) == type(False):
        minarrs = []
        maxarrs = []
        for i in range(num_feats):
            minarrs.append(np.min(arr[:,:,i]))
            maxarrs.append(np.max(arr[:,:,i]))



    for i in range(num_feats):
        scaled_arr[:,:,i], _, _ = scale_vector(  
                                       arr[:,:,i], 
                                       minarrs[i], 
                                       maxarrs[i]
                                       )
    return scaled_arr, minarrs, maxarrs

def scale_Y(arr, minarrs=False, maxarrs=False):
    '''
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
    '''
    num_feats = np.shape(arr)[-1]
    scaled_arr = copy.deepcopy(arr)

    if not minarrs:
        minarrs = []
        maxarrs = []
        for i in range(num_feats):
            minarrs.append(np.min(arr[:,i]))
            maxarrs.append(np.max(arr[:,i]))



    for i in range(num_feats):
        scaled_arr[:,i], _, _ = scale_vector(  
                                       arr[:,i], 
                                       minarrs[i], 
                                       maxarrs[i]
                                       )
    return scaled_arr, minarrs, maxarrs

def scale_vector(arr, minarr, maxarr):
    '''
    Min-max scale a vector from -1 to +1.  

    Parameters:
        arr (numpy array)
            --  1D vector that we wish to scale. 
        minarrs and maxarrs (boolean or list)
            --  If minarrs is set to False, then we calculate the minimum and
                maximum of each feature and scale between -1 and 1. If minarrs
                and maxarrs are given as lists, we use them to min-max scale
                each feature of the array following:

                arr_scaled = 
                    [2*(arr - min(arr) / 
                    (max(arr) - min(arr))] - 1
    Returns:
        scaled_arr (numpy array)
            --  The scaled vector.
        minarrs and maxarrays (list)
            --  Lists of the minimum and maximum values of the arrays BEFORE
                we scale them.
    '''

    scaled_X = (arr - minarr) / (maxarr - minarr)
    scaled_X =  2*scaled_X - 1

    return scaled_X, minarr, maxarr

def unscale_vector(arr, minarr, maxarr):
    '''
    Unscale a vector that has previously been min-max scaled.  

    Parameters:
        arr (numpy array)
            --  1D vector that we wish to unscale. 
        minarrs and maxarrs (floats)
            --  The maximum and minimum that we wish to unscale according to.
    Returns:
        unscaled_arr (numpy array)
            --  The unscaled array
    '''

    arr = (arr+1)/2
    unscaled_arr = arr*(maxarr - minarr) + minarr
    return unscaled_arr

def unscale_X(arr, minarrs, maxarrs):
    '''
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
    '''
    unscaled_arr = copy.deepcopy(arr)
    for i in range(12):
        minarr = minarrs[i]
        maxarr = maxarrs[i]
        unscaled_arr[:,i] = unscale_vector(
                                           arr[:,i], 
                                           minarrs[i], 
                                           maxarrs[i]
                                           )
    return unscaled_arr

def unscale_3D_arr(arr, minarrs, maxarrs):
    '''
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
    '''
    unscaled_arr = copy.deepcopy(arr)
    for i in range(12):
        minarr = minarrs[i]
        maxarr = maxarrs[i]
        unscaled_arr[:,:,i] = unscale_vector(
                                           arr[:,:,i], 
                                           minarrs[i], 
                                           maxarrs[i]
                                           )
    return unscaled_arr

def unscale_predicted(predicted_current, xmins, xmaxs):                                 
    unscaled = []
    for i in range(4):
        unscaled_vec = unscale_vector(
                                   predicted_current[:,i],
                                   xmins[i],
                                   xmaxs[i]
                                   )
        unscaled.append(unscaled_vec)
    return np.array(unscaled)

###############################################################################
#
# Functions for loading data
#
###############################################################################

def concat_X_and_Y(X,Y):
    X_temp = copy.deepcopy(X)
    X_temp = np.reshape(np.transpose(
                                      X_temp, 
                                      (0, 2, 1)), 
                                      [np.shape(X_temp)[0], 
                                      NNv.num_IdVg*NNv.n_points*4]
                                      )

    Z = np.concatenate([Y, X_temp[:,0:2*NNv.num_IdVg*NNv.n_points]], axis=1)
    return Z

def interpolate_data(data, new_V, logscale = False):
    V = data[0]
    Id = data[1]
    interp_func = interp1d(V, Id)
    new_y = interp_func(new_V)
    return(new_y) 

def process_folder(
                   dirname, 
                   V, 
                   n_points, 
                   num_IdVg, 
                   num_feats, 
                   minval, 
                   ):
    X_unscaled, Y_unscaled = extract_folder(
                                            dirname, 
                                            V, 
                                            n_points, 
                                            num_IdVg, 
                                            num_feats, 
                                            minval
                                            )
    
    current_indices = []
    deriv_indices = []
    for i in range(NNv.num_IdVg):
        current_indices.append(0+i*4) 
        current_indices.append(1+i*4) 
        deriv_indices.append(2+i*4) 
        deriv_indices.append(3+i*4) 

    X_unscaled = np.concatenate([
                                 X_unscaled[:,:, current_indices], 
                                 X_unscaled[:,:, deriv_indices]
                                 ], 
                                 axis=-1)

    X, Xmins, Xmaxs  = scale_X(X_unscaled)
    Y, Ymins, Ymaxs  = scale_Y(Y_unscaled)    
    np.savetxt(dir_path + '/Xscaling.dat', np.array([Xmins, Xmaxs]))
    np.savetxt(dir_path + '/Yscaling.dat', np.array([Ymins, Ymaxs]))
    return X, Y

def extract_folder(dir_name, V, n_points, num_IdVg, num_feats, minval):
    subdirs = sorted(glob.glob(dir_name + '/*'))    
    counter = 0
    crit_mass = 10000
    tick = time.time()
    X_array_final = []
    Y_array_final = []
    X_array = []
    Y_array = []
    num_saves = 0
    for subdir in subdirs:
        if counter%crit_mass == 0 and counter > 1:
            num_saves += 1
            X_array_final.append(X_array)
            Y_array_final.append(Y_array)
            X_array = []
            Y_array = []
            num_saves += 1

        counter += 1 # for keeping track of number of processed daa
        
        if counter % 250 == 0:
            tock = time.time()
            print(counter, tock - tick, np.shape(X_array))
            tick = tock


        y, variable_names = build_y_array(subdir + '/variables.csv')
        x = np.array([])
        
        subdir_files = glob.glob(subdir + '/*')
        subdir_files = sorted(subdir_files)
        Flag = False
        Id = 0
        
        for filename in subdir_files:
            if (not 'IdVg' in filename and not 'IdVd' in filename):
                continue
            try:
                x = build_x_array(x, filename, V, minval)

            except Exception as e:
                print(e)
                Flag = True
                continue

        if not x.size == (num_feats*num_IdVg*n_points) or Flag:
            continue

        x = np.array(x)
        x = np.reshape(x, (num_feats*num_IdVg, n_points))
        x = x.T
        x = np.reshape(x, (1, n_points, num_feats*num_IdVg))
        X_array.append(x)
        Y_array.append(y)

        with open(dir_path + '/variable_names.txt', 'w') as file:
            for string in variable_names:
                file.write(string + '\n')


    X_array_final.append(X_array)
    Y_array_final.append(Y_array)

    X = np.concatenate(X_array_final, axis=0)
    Y = np.concatenate(Y_array_final, axis=0)

    X = X.reshape(X.shape[0], X.shape[2], X.shape[3])
    Y = Y.reshape(Y.shape[0], Y.shape[2])
    
    return X, Y

def build_x_array(x, filename, V, minval):
    # sentaurus and hemt simulations are formatted differently, so we need to
    # use different calls to load the data
    if NNv.simtype == 'sentaurus':
        data = np.loadtxt(filename, skiprows = 1, delimiter = ',').T
    elif NNv.simtype == 'hemt':
        data = np.loadtxt(filename, usecols = range(2)).T
        data[1] = np.abs(data[1])
 
    Id = interpolate_data([data[0], data[1]], V, logscale = False)
    Id_log = interpolate_data(
                              [data[0], np.log10(np.abs(data[1]))], 
                              V, 
                              logscale = False
                              )
    
    indices = np.where(Id < minval)
    Id[indices] = minval
    indices = np.where(Id_log < np.log10(minval))
    Id_log[indices] = np.log10(minval)
    
    Id_grad = np.gradient(Id,V)
    Id_grad_log = np.gradient(Id_log,V)

    x = np.concatenate((x, 
                        Id, 
                        Id_log, 
                        Id_grad, 
                        Id_grad_log,
                        ))
    return x

def build_y_array(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='=')
        y = [float(row[1]) for row in reader]

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='=')
        variable_names = [str(row[0]) for row in reader]

    y = np.array(y, dtype = 'float64')
    y = np.reshape(y, (1, np.size(y)))
    return y, variable_names

def load_exp(filename, sheetname, V, gateVcol = 'GateV', Idcol = 'DrainI', start = 1102, stop = 1854, skip = 10, W = 2):
    df = pd.read_excel(filename, sheet_name=sheetname, usecols=[gateVcol, Idcol])
    Vg = np.array(df["GateV"])[start:stop]
    Id = np.array(df["DrainI"])[start:stop]/W
    Id_int = interpolate_data([Vg, Id], V)
    Id_int_log = interpolate_data([Vg, np.log10(np.abs(Id))], V)
    plt.plot(Vg[::skip], Id[::skip], color = 'k', marker = 'o', ls = 'None')
    plt.plot(V, Id_int, color = 'r', ls = '--')
    plt.savefig(filename.replace('.xlsx', '.png'))
    plt.close()

    plt.plot(Vg[::skip], np.log10(Id[::skip]), color = 'k', marker = 'o', ls = 'None')
    plt.plot(V, Id_int_log, color = 'r', ls = '--')
    plt.savefig(filename.replace('.xlsx', '_log.png'))
    plt.close()

    return (Id_int, Id_int_log)

def process_device(dev):
    # process an experimental device 

    dev_100_filename =  dir_path + '/exp_data/' + dev + '_100mVds.xlsx'
    dev_1000_filename = dir_path + '/exp_data/' + dev + '_1Vds.xlsx'
    Id_100, Id_100_log = load_exp(dev_100_filename, '{}_100mVds'.format(dev), V)
    Id_1000, Id_1000_log = load_exp(dev_1000_filename, '{}_1Vds'.format(dev), V)

    Id_100_grad = np.gradient(Id_100, V)
    Id_100_log_grad = np.gradient(Id_100_log, V)
    Id_100_grad2 = np.gradient(Id_100_grad, V)
    Id_100_log_grad2 = np.gradient(Id_100_log_grad, V)

    Id_1000_grad = np.gradient(Id_1000, V)
    Id_1000_log_grad = np.gradient(Id_1000_log, V)
    Id_1000_grad2 = np.gradient(Id_1000_grad, V)
    Id_1000_log_grad2 = np.gradient(Id_1000_log_grad, V)

    x_input = np.array([
    Id_100,
    Id_100_log,
    Id_100_grad,
    Id_100_log_grad,
    Id_100_grad2,
    Id_100_log_grad2,
    Id_1000,
    Id_1000_log,
    Id_1000_grad,
    Id_1000_log_grad,
    Id_1000_grad2,
    Id_1000_log_grad2,
    ])

    return x_input

def load_exp(filename, sheetname, V, gateVcol = 'GateV', Idcol = 'DrainI', start = 1102, stop = 1854, skip = 1, W = 1):
    df = pd.read_excel(filename, sheet_name=sheetname, usecols=[gateVcol, Idcol])
    Vg = np.array(df["GateV"])[start:stop]
    Id = np.array(df["DrainI"])[start:stop]/W
    Id_int = interpolate_data([Vg, Id], V)
    Id_int_log = interpolate_data([Vg, np.log10(np.abs(Id))], V)
    plt.plot(Vg[::skip], Id[::skip], color = 'k', marker = 'o', ls = 'None')
    plt.plot(V, Id_int, color = 'r', ls = '--')
    plt.savefig(filename.replace('.xlsx', '.png'))
    plt.close()

    plt.plot(Vg[::skip], np.log10(Id[::skip]), color = 'k', marker = 'o', ls = 'None')
    plt.plot(V, Id_int_log, color = 'r', ls = '--')
    plt.savefig(filename.replace('.xlsx', '_log.png'))
    plt.close()

    return (Id_int, Id_int_log)

def process_device(dev, V):
    dev_100_filename =  dir_path + '/exp_data/' + dev + '_100mVds.xlsx'
    dev_1000_filename = dir_path + '/exp_data/' + dev + '_1Vds.xlsx'
    Id_100, Id_100_log = load_exp(dev_100_filename, '{}_100mVds'.format(dev), V)
    Id_1000, Id_1000_log = load_exp(dev_1000_filename, '{}_1Vds'.format(dev), V)
    
    minval = 1e-12

    indices = np.where(Id_100 < minval)                                             
    Id_100[indices] = minval
    Id_100_log[indices] = np.log10(minval)
    indices = np.where(Id_1000 < minval)                                             
    Id_1000[indices] = minval
    Id_1000_log[indices] = np.log10(minval)

    Id_100_grad = np.gradient(Id_100, V)
    Id_100_log_grad = np.gradient(Id_100_log, V)
    Id_100_grad2 = np.gradient(Id_100_grad, V)
    Id_100_log_grad2 = np.gradient(Id_100_log_grad, V)

    Id_1000_grad = np.gradient(Id_1000, V)
    Id_1000_log_grad = np.gradient(Id_1000_log, V)
    Id_1000_grad2 = np.gradient(Id_1000_grad, V)
    Id_1000_log_grad2 = np.gradient(Id_1000_log_grad, V)

    x_input = np.array([
    Id_100,
    Id_100_log,
    Id_100_grad,
    Id_100_log_grad,
    Id_100_grad2,
    Id_100_log_grad2,
    Id_1000,
    Id_1000_log,
    Id_1000_grad,
    Id_1000_log_grad,
    Id_1000_grad2,
    Id_1000_log_grad2,
    ])

    return x_input

def process_exp(data_exp_100, data_exp_1000, new_V, minval):
    Vg_100, Id_100 = data_exp_100[0], data_exp_100[1]
    Vg_1000, Id_1000 = data_exp_1000[0], data_exp_1000[1]

    Id_100_log = interpolate_data([Vg_100, np.log10(Id_100)], new_V)
    Id_1000_log = interpolate_data([Vg_1000, np.log10(Id_1000)], new_V)
    Id_100 = interpolate_data([Vg_100, Id_100], new_V)
    Id_1000 = interpolate_data([Vg_1000, Id_1000], new_V)

    indices = np.where(Id_100 < minval)                                             
    Id_100[indices] = minval
    Id_100_log[indices] = np.log10(minval)
    indices = np.where(Id_1000 < minval)                                             
    Id_1000[indices] = minval
    Id_1000_log[indices] = np.log10(minval)

    Id_100_grad = np.gradient(Id_100, new_V)
    Id_100_log_grad = np.gradient(Id_100_log, new_V)
    Id_100_grad2 = np.gradient(Id_100_grad, new_V)
    Id_100_log_grad2 = np.gradient(Id_100_log_grad, new_V)

    Id_1000_grad = np.gradient(Id_1000, new_V)
    Id_1000_log_grad = np.gradient(Id_1000_log, new_V)
    Id_1000_grad2 = np.gradient(Id_1000_grad, new_V)
    Id_1000_log_grad2 = np.gradient(Id_1000_log_grad, new_V)

    x_input = np.array([
    Id_100,
    Id_100_log,
    Id_100_grad,
    Id_100_log_grad,
    Id_100_grad2,
    Id_100_log_grad2,
    Id_1000,
    Id_1000_log,
    Id_1000_grad,
    Id_1000_log_grad,
    Id_1000_grad2,
    Id_1000_log_grad2,
    ])

    return x_input
