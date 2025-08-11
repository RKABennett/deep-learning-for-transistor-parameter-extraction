import copy
import json
import os
import random
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .utils_misc import unscale_vector

dir_path = os.path.dirname(os.path.abspath(sys.argv[0])) 
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(root_dir, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

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

    Y_pred = np.array(model_inverse.predict(X_test))[:, 0:cfg["data"]["num_params"]]
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

    np.savetxt(save_loc + '/{}'.format(error_filename), errors)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(
             errors, 
             density = False,
             edgecolor = 'k', 
             linewidth=0.25,
             color = '#1048a2'
             )


    plt.tight_layout()
    plt.savefig(save_loc + '/R2_histogram.png')
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
            plt.savefig(save_loc + '/inverse_plot_{}.png'.format(i))
            plt.close()

    if save_fits:
        data_folder = save_loc + '/fits_inverse'
        actual_fits = save_loc + '/fits_inverse/actual'.format(fit_name)
        pred_fits =   save_loc + '/fits_inverse/pred'.format(fit_name)

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
            actual_filename = 'error={:.12f}_actual.dat'.format(errors[i])
            pred_filename = 'error={:.12f}_pred.dat'.format(errors[i])
            np.savetxt(actual_fits + '/' + actual_filename, actuals)
            np.savetxt(pred_fits + '/' + pred_filename, preds)
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

    np.savetxt(save_file_loc + '/{}'.format(error_filename), errors)

    ###########################################################################
    #
    # Save fits to file
    #
    ###########################################################################
    data_folder = save_file_loc + '/fits_forward'
    actual_fits = save_file_loc + '/fits_forward/{}_actual'.format(fit_name)
    pred_fits =   save_file_loc + '/fits_forward/{}_pred'.format(fit_name)

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
            fig, axs = plt.subplots(cfg["data"]["num_IdVg"], 2, figsize=(12, 10))
            for j in range(np.shape(X_test)[2]):
                row, col = divmod(j, 2)  # Determine row and column for the subplot
                axs[row, col].plot(X_test[i, :, j], color='k', marker='o', ls='None')
                axs[row, col].plot(X_pred[i, :, j], 'r', ls='--')
    
            plt.tight_layout()
            plt.savefig(save_file_loc + '/forward_plot_{}.png'.format(i))
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


