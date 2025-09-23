import copy
import json
import os
from pathlib import Path 
import random
import sys

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .utils_misc import unscale_vector

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(root_dir, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

def test_model_inverse_current(
    X_test,
    Y_test,
    model_forward,
    model_inverse,
    Xscaling,
    Yscaling,
    error_metric,
    error_filename,
    save_loc,
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
   
    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]

    Ymin = Yscaling[0,:]
    Ymax = Yscaling[1,:]

    Z_pred = np.array(model_inverse.predict(X_test))
    X_pred = np.array(model_forward.predict(Z_pred[:, 0:cfg["data"]["num_params"]]))


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
        for j in range(cfg["data"]["num_IdVg"] * 2):
            if deriv_error:
                error += error_metric(
                                      np.gradient(X_test[i,:,j]), 
                                      np.gradient(X_pred[i,:,j])
                                      )
            else:
                error += error_metric(X_test[i,:,j], X_pred[i,:,j])
        errors.append(error/(cfg["data"]["num_IdVg"]*2))

    np.savetxt(Path(save_loc) / error_filename, errors)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(
             errors, 
             density = False,
             edgecolor = 'k', 
             linewidth=0.25,
             color = '#1048a2'
             )


    plt.tight_layout()
    plt.savefig(Path(save_loc) / 'R2_histogram.png')
    plt.close()
    
    ###########################################################################
    #
    # Plot actual vs. predicted current
    #
    ########################################################################### 
    if plot:
        for i in plot_range:
            print('Plot number {}'.format(i))
            fig, axs = plt.subplots(2, 2, figsize=(12, 4*cfg["data"]["num_IdVg"]))
            for j in range(cfg["data"]["num_IdVg"]*2):
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
            plt.savefig(Path(save_loc) / f'inverse_plot_{i}.png')
            plt.close()

    if save_fits:
        data_folder = Path(save_loc) / 'fits_inverse'
        actual_fits = Path(save_loc) / 'fits_inverse' / 'actual'
        pred_fits = Path(save_loc) / 'fits_inverse' / 'pred'

        for dir_name in [data_folder, actual_fits, pred_fits]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        for i in range(np.shape(X_test)[0]):
            actuals = []
            preds = []
            for j in range(cfg["data"]["num_IdVg"]*2):
                Id_actual = X_test[i,:, j]
                Id_pred   = X_pred[i,:, j]
                actuals.append(Id_actual)
                preds.append(Id_pred)
            actual_filename = f'error={errors[i]:.12f}_actual.dat'
            pred_filename = f'error={errors[i]:.12f}_pred.dat'
            np.savetxt(Path(actual_fits) / actual_filename, actuals)
            np.savetxt(Path(pred_fits) / pred_filename, preds)
    return X_test, X_pred, errors

def test_model_forward(
        X_test,
        Y_test,
        model_forward,
        Xscaling,
        Yscaling,
        error_metric,
        error_filename,
        save_file_loc,
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

    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]

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

    np.savetxt(Path(save_file_loc) / error_filename, errors)
    ###########################################################################
    #
    # Save fits to file
    #
    ###########################################################################
    data_folder = Path(save_file_loc) / 'fits_forward'
    actual_fits = Path(save_file_loc) / 'fits_forward' / f'{fit_name}_actual'
    pred_fits = Path(save_file_loc) / 'fits_forward' / f'{fit_name}_pred'
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

        actual_filename = f'error={errors[i]:.12f}_actual.dat'
        pred_filename = f'error={errors[i]:.12f}_pred.dat'
        np.savetxt(Path(actual_fits) / actual_filename, actuals)
        np.savetxt(Path(pred_fits) / pred_filename, preds)

    ###########################################################################
    #
    # Optionally plot
    #
    ###########################################################################
    if plot:
        for i in plot_range:
            print('Plot number {}'.format(i))
            fig, axs = plt.subplots(cfg["data"]["num_IdVg"], 2, figsize=(12, 10))
            for j in range(np.shape(X_test)[2]):
                row, col = divmod(j, 2)  # Determine row and column for the subplot
                axs[row, col].plot(X_test[i, :, j], color='k', marker='o', ls='None')
                axs[row, col].plot(X_pred[i, :, j], 'r', ls='--')
    
            plt.tight_layout()
            plt.savefig(Path(save_file_loc) / f'forward_plot_{i}.png')
            plt.close()

    return X_test, X_pred, errors

