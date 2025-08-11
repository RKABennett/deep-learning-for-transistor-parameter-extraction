import copy
import csv
import glob
import json
import numpy as np
import os
from scipy.interpolate import interp1d
import sys
import time


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(root_dir, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))


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
                   processed_data_loc,
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
    np.savetxt(processed_data_loc + '/Xscaling.dat', np.array([Xmins, Xmaxs]))
    np.savetxt(processed_data_loc + '/Yscaling.dat', np.array([Ymins, Ymaxs]))
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
