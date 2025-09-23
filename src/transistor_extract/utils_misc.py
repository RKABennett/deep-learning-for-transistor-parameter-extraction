import copy
import csv
import glob
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import interp1d
import sys
import time


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(root_dir, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

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
                                      cfg["data"]["num_IdVg"]*cfg["data"]["n_points"]*4]
                                      )

    Z = np.concatenate([Y, X_temp[:,0:2*cfg["data"]["num_IdVg"]*cfg["data"]["n_points"]]], axis=1)
    return Z

def interpolate_data(data, new_V, logscale = False):
    V = data[0]
    Id = data[1]
    interp_func = interp1d(V, Id)
    new_y = interp_func(new_V)
    return(new_y) 

def process_folder(
                   dirname, 
                   working_dir,
                   processed_data_loc,
                   V, 
                   n_points, 
                   num_IdVg, 
                   num_feats, 
                   minval,
                   print_progress = False,
                   print_frequency = 1000
                   ):
    X_unscaled, Y_unscaled = extract_folder(
                                            dirname,
                                            working_dir,
                                            V, 
                                            n_points, 
                                            num_IdVg, 
                                            num_feats, 
                                            minval,
                                            print_progress = print_progress,
                                            print_frequency = print_frequency
                                            )
    
    current_indices = []
    deriv_indices = []
    for i in range(cfg["data"]["num_IdVg"]):
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
    np.savetxt(Path(processed_data_loc) / 'Xscaling.dat', np.array([Xmins, Xmaxs]))
    np.savetxt(Path(processed_data_loc) / 'Yscaling.dat', np.array([Ymins, Ymaxs]))
    return X, Y

def extract_folder(
                   dir_name,
                   working_dir,
                   V, 
                   n_points, 
                   num_IdVg, 
                   num_feats, 
                   minval,
                   print_progress = False,
                   print_frequency = 1000
                   ):

    subdirs = sorted(str(p) for p in Path(dir_name).glob('*'))    
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
        
        
        if print_progress and counter % print_frequency == 0:
            print('Processed {} devices. Current array shape: {}' .format(counter,  np.shape(X_array)))


        
        y, variable_names = build_y_array(Path(subdir) / 'variables.csv')
        x = np.array([])
        
        subdir_files = sorted(str(p) for p in Path(subdir).glob('*'))
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

        with open(Path(working_dir) / 'variable_names.txt', 'w') as file:
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
    if cfg["data"]["simtype"] == 'sentaurus':
        data = np.loadtxt(filename, skiprows=1, delimiter=',').T
    elif cfg["data"]["simtype"] == 'hemt':
        data = np.loadtxt(filename, usecols=range(2)).T
        data[1] = np.abs(data[1])

    Id = interpolate_data([data[0], data[1]], V, logscale=False)
    Id_log = interpolate_data(
                              [data[0], np.log10(np.abs(data[1]))],
                              V,
                              logscale=False
                              )

    indices = np.where(Id < minval)
    Id[indices] = minval
    indices = np.where(Id_log < np.log10(minval))
    Id_log[indices] = np.log10(minval)

    Id_grad = np.gradient(Id, V)
    Id_grad_log = np.gradient(Id_log, V)

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

def process_device(dev, V, working_dir):
    dev_100_filename = Path(working_dir) / 'exp_data' / f'{dev}_100mVds.xlsx'
    dev_1000_filename = Path(working_dir) / 'exp_data' / f'{dev}_1Vds.xlsx'
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

###############################################################################
#
# Functions used only in the notebook demo
#
###############################################################################

def plot_variables(model_inverse, plot_idx, X_test, Y_test, Xscaling, Yscaling):
    fontsize = 9
    mpl.rcParams.update({'font.size': fontsize})

    blue = '#19546d'
    red = '#bd2b49'
    purple = '#192a6d'


    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]

    Ymins = Yscaling[0,:]
    Ymaxs = Yscaling[1,:]

    Y_pred = np.array(model_inverse.predict(X_test))[:, 0:cfg["data"]["num_params"]]

    ticks = [
              [0, 15, 30], 
              [0, 125, 250, 375, 500], 
              [0,0.5,1],
              [0,1,2,3],
              [0, 50, 100, 150, 200],
              [0, 50, 100, 150, 200],
              [0, 1, 2, 3],
              [50, 175, 300],
              ]
    subset = range(250)

    variables = [
                'Mobility (cm$^2$  V$^{-1}$ s $^{-1}$)',
                'Schottky barrier height (meV)',
                'Effective density of states ($\\times 10^{13}$ cm$^{-2}$)',
                'Peak donor density ($\\times$ 10$^{13}$ cm$^{-2}$ eV$^{-1}$)',
                'Donor energy mid (meV below conduction band edge)', 
                'Donor energy width (meV)',
                'Peak acceptor band tail density ($\\times$ 10$^{13}$ cm$^{-2}$ eV$^{-1}$)',
                'Acceptor band tail energy width (meV)',
                ]

    fig, axs = plt.subplots(1,2, figsize = (3.5, 2.25))
    plt.subplots_adjust(left = 0.13, top = 0.71, right = 0.9, bottom = 0.175, hspace = 0.5, wspace = 0.7)
    Ymin = Ymins[plot_idx]
    Ymax = Ymaxs[plot_idx]

    if plot_idx in [2]:
        Ymin*=6.15e-8 / 1e13
        Ymax*=6.15e-8 / 1e13
    elif plot_idx in [3]:
        Ymin/= 1e13
        Ymax/=1e13
    if plot_idx in [6]:
        Ymin*=6.15e-8 / 1e13
        Ymax*=6.15e-8 / 1e13
    elif plot_idx in [4,5,7]:
        Ymin *= 1000
        Ymax *= 1000

    Y_test[:,plot_idx] = unscale_vector(Y_test[:,plot_idx], Ymin, Ymax)
    Y_pred[:,plot_idx] = unscale_vector(Y_pred[:,plot_idx], Ymin, Ymax)
    if plot_idx == 1:
        Y_test[:,plot_idx] = 5000 - 1000*Y_test[:,plot_idx]
        Y_pred[:,plot_idx] = 5000 - 1000*Y_pred[:,plot_idx]
        Ymin = 0
        Ymax = 500

    axs[0].plot(
            Y_test[subset,plot_idx], 
            Y_pred[subset,plot_idx], 
            marker = 'o', 
            ls = 'None',
            markersize = 4,
            color = 'k',
            markerfacecolor = purple,
            markeredgewidth = 0.4
            )

    axs[0].plot(
            [-10000, 10000], 
            [-10000, 10000], 
            color = red, 
            ls = '--'
            )

    axs[0].set_xlim([Ymin, Ymax])
    axs[0].set_ylim([Ymin, Ymax])
    axs[0].set_xticks(ticks[plot_idx])
    axs[0].set_yticks(ticks[plot_idx])
    errors = (Y_test[:,plot_idx] - Y_pred[:,plot_idx])

    MAE = np.median(np.abs(errors))
    std = np.std(errors)
    binmin = -4*std
    binmax = 4*std
    binwidth = (binmax - binmin)/25
    bins = np.arange(binmin, binmax, binwidth)
    axs[1].hist(
                errors, 
                bins = bins,
                color = purple,
                edgecolor = 'k',
                linewidth = 0.15
                )

    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Counts')

    fig.text(0.5, 0.88, variables[plot_idx], ha='center', fontsize=10.5)     
    print('\n \n Median absolute error = {} \n Standard deviation of error = {}'.format(
                    round(MAE, 3),
                    round(std, 3)))

    plt.show()

def plot_inverse(quantile, errors, data_pred_base, data_actual_base):
    num_entries = len(errors)
    target = int(num_entries*quantile)
    errors = sorted(errors)
    target_error = errors[target]
    
    data_pred = np.loadtxt(data_pred_base.format(target_error))
    data_actual = np.loadtxt(data_actual_base.format(target_error))

    Id_100_pred = data_pred[0]
    Id_100_log_pred = data_pred[1]
    Id_1000_pred = data_pred[2]
    Id_1000_log_pred = data_pred[3]
    
    Id_100_actual = data_actual[0]
    Id_100_log_actual = data_actual[1]
    Id_1000_actual = data_actual[2]
    Id_1000_log_actual = data_actual[3]
    
    ###############################################################################
    #
    # Plot data
    #
    ###############################################################################
    
    fig, ax1 = plt.subplots(1,1)
    ax2 = ax1.twinx()
    
    start, stop, skip = 0, 32, 3
    zorder_pred = 100001
    zorder_actual = 10000
    actual_OLcolor = 'k'
    actual_Fcolor_1 = '#4dadd6'
    pred_color_1 = '#19546d'
    actual_Fcolor_01 = '#d64d69'
    pred_color_01 = '#6d192a'
    
    
    scale = 10**6 # A/um to uA/um conversion factor
    V = np.linspace(
                    cfg["data"]["Vmin"],
                    cfg["data"]["Vmax"],
                    cfg["data"]["n_points"]
                    )
    
    
    ax2.plot(
            V[start:stop:skip], 
            scale*Id_100_actual[start:stop:skip],
            marker='o',
            color=actual_OLcolor,
            markerfacecolor = actual_Fcolor_01,
            ls='None',
            label='Vds=0.1, Actual',
            zorder = zorder_actual
            )
    ax2.plot(
            V, 
            scale*Id_100_pred,
            marker='None',
            color=pred_color_01, 
            ls='-',
            label='Vds=0.1, Pred',
            zorder = zorder_pred
           )
    
    # Linear scale, Vds = 1.0
    ax2.plot(
            V[start:stop:skip], 
            scale*Id_1000_actual[start:stop:skip],
            marker='s',
            color=actual_OLcolor,
            markerfacecolor = actual_Fcolor_1,
            ls='None',
            label='Vds=1.0, Actual',
            zorder = zorder_actual,
            )
    
    ax2.plot(
            V, 
            scale*Id_1000_pred,
            marker='None',
            color=pred_color_1, 
            ls='-',
            label='Vds=1.0, Pred',
            zorder = zorder_pred
            )
    
    # Log scale, Vds = 0.1
    ax1.semilogy(
            V[start:stop:skip], 
            scale*np.power(10, Id_100_log_actual)[start:stop:skip],
            marker='o',
            color=actual_OLcolor,
            markerfacecolor = actual_Fcolor_01,
            ls='None',
            label='Log Vds=0.1, Actual',
            zorder = zorder_actual
            )
    
    ax1.semilogy(
            V, 
            scale*np.power(10, Id_100_log_pred),
            marker='None',
            color=pred_color_01, 
            ls='-',
            label='Log Vds=0.1, Pred',
            zorder = zorder_pred
            )
    
    # Log scale, Vds = 1.0
    ax1.semilogy(
            V[start:stop:skip], 
            scale*np.power(10, Id_1000_log_actual)[start:stop:skip],
            marker='s',
            color=actual_OLcolor,
            markerfacecolor = actual_Fcolor_1,
            ls='None',
            label='Log Vds=0.1, Actual',
            zorder = zorder_actual
            )
    
    ax1.semilogy(
            V, 
            scale*np.power(10, Id_1000_log_pred),
            marker='None',
            color=pred_color_1, 
            ls='-',
            label='Log Vds=0.1, Pred',
            zorder = zorder_pred
            )
    
    plt.show()

def plot_forward(
                 quantile, 
                 errors,
                 data_pred_base, 
                 data_actual_base
                 ):
    
    num_entries = len(errors)
    target = int(num_entries*quantile)
    errors = sorted(errors)
    target_error = errors[target]
    
    data_pred = np.loadtxt(data_pred_base.format(target_error))
    data_actual = np.loadtxt(data_actual_base.format(target_error))
    
    V = np.linspace(-5.9, 49.9, 32)
    
    Id_100_pred = data_pred[0]
    Id_100_log_pred = data_pred[1]
    Id_1000_pred = data_pred[2]
    Id_1000_log_pred = data_pred[3]
    
    Id_100_actual = data_actual[0]
    Id_100_log_actual = data_actual[1]
    Id_1000_actual = data_actual[2]
    Id_1000_log_actual = data_actual[3]

    scale = 10**6
    
    fig, ax1 = plt.subplots(1,1)
    ax2 = ax1.twinx()
    
    start, stop, skip = 0, 32, 3
    
    zorder_pred = 10001
    zorder_actual = 10000
    
    actual_OLcolor = 'k'
    
    actual_Fcolor_1 = '#4dadd6'
    pred_color_1 = '#19546d'
    
    actual_Fcolor_01 = '#d64d69'
    pred_color_01 = '#6d192a'
    
    # Linear scale, Vds = 0.1
    ax2.plot(
        V[start:stop:skip], 
        scale*Id_100_actual[start:stop:skip],
        marker='o',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_01,
        ls='None',
        label='Vds=0.1, Actual',
        zorder = zorder_actual
        )
    ax2.plot(
        V, 
        scale*Id_100_pred,
        marker='None',
        color=pred_color_01, 
        ls='-',
        label='Vds=0.1, Pred',
        zorder = zorder_pred
       )
    
    # Linear scale, Vds = 1.0
    ax2.plot(
        V[start:stop:skip], 
        scale*Id_1000_actual[start:stop:skip],
        marker='s',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_1,
        ls='None',
        label='Vds=1.0, Actual',
        zorder = zorder_actual,
        )
    
    ax2.plot(
        V, 
        scale*Id_1000_pred,
        marker='None',
        color=pred_color_1, 
        ls='-',
        label='Vds=1.0, Pred',
        zorder = zorder_pred
        )
    
    # Log scale, Vds = 0.1
    ax1.semilogy(
        V[start:stop:skip], 
        scale*np.power(10, Id_100_log_actual)[start:stop:skip],
        marker='o',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_01,
        ls='None',
        label='Log Vds=0.1, Actual',
        zorder = zorder_actual
        )
    
    ax1.semilogy(
        V, 
        scale*np.power(10, Id_100_log_pred),
        marker='None',
        color=pred_color_01, 
        ls='-',
        label='Log Vds=0.1, Pred',
        zorder = zorder_pred
        )
    
    # Log scale, Vds = 1.0
    ax1.semilogy(
        V[start:stop:skip], 
        scale*np.power(10, Id_1000_log_actual)[start:stop:skip],
        marker='s',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_1,
        ls='None',
        label='Log Vds=0.1, Actual',
        zorder = zorder_actual
        )
    
    ax1.semilogy(
        V, 
        scale*np.power(10, Id_1000_log_pred),
        marker='None',
        color=pred_color_1, 
        ls='-',
        label='Log Vds=0.1, Pred',
        zorder = zorder_pred
        )
    
    plt.show()
    plt.close()
