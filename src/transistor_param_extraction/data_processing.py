"""
Data processing and loading functions for transistor parameter extraction.

This module contains functions for loading, processing, interpolating, and scaling
data from both simulation and experimental sources.
"""

import os
import sys
import copy
import csv
import glob
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from . import NN_variables as NNv
from .scaling import scale_X, scale_Y, unscale_vector

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))


def concat_X_and_Y(X, Y):
    """
    Concatenate X and Y arrays for inverse neural network training.

    Parameters:
        X (numpy array): Input current-voltage data
        Y (numpy array): Output parameter data

    Returns:
        Z (numpy array): Concatenated array for training
    """
    X_temp = copy.deepcopy(X)
    X_temp = np.reshape(
        np.transpose(X_temp, (0, 2, 1)),
        [np.shape(X_temp)[0], NNv.num_IdVg * NNv.n_points * 4],
    )

    Z = np.concatenate([Y, X_temp[:, 0 : 2 * NNv.num_IdVg * NNv.n_points]], axis=1)
    return Z


def interpolate_data(data, new_V, logscale=False):
    """
    Interpolate current-voltage data to a new voltage grid.

    Parameters:
        data (tuple): (voltage, current) data
        new_V (array): New voltage grid
        logscale (bool): Whether to use log scaling

    Returns:
        new_y (array): Interpolated current data
    """
    V = data[0]
    Id = data[1]
    interp_func = interp1d(V, Id)
    new_y = interp_func(new_V)
    return new_y


def process_folder(dirname, V, n_points, num_IdVg, num_feats, minval):
    """
    Process a folder containing simulation data.

    Parameters:
        dirname (str): Directory containing data
        V (array): Voltage grid
        n_points (int): Number of voltage points
        num_IdVg (int): Number of Id-Vg curves
        num_feats (int): Number of features per curve
        minval (float): Minimum current value

    Returns:
        X, Y (tuple): Processed and scaled data arrays
    """
    X_unscaled, Y_unscaled = extract_folder(
        dirname, V, n_points, num_IdVg, num_feats, minval
    )

    current_indices = []
    deriv_indices = []
    for i in range(NNv.num_IdVg):
        current_indices.append(0 + i * 4)
        current_indices.append(1 + i * 4)
        deriv_indices.append(2 + i * 4)
        deriv_indices.append(3 + i * 4)

    X_unscaled = np.concatenate(
        [X_unscaled[:, :, current_indices], X_unscaled[:, :, deriv_indices]], axis=-1
    )

    X, Xmins, Xmaxs = scale_X(X_unscaled)
    Y, Ymins, Ymaxs = scale_Y(Y_unscaled)
    np.savetxt(dir_path + "/Xscaling.dat", np.array([Xmins, Xmaxs]))
    np.savetxt(dir_path + "/Yscaling.dat", np.array([Ymins, Ymaxs]))
    return X, Y


def extract_folder(dir_name, V, n_points, num_IdVg, num_feats, minval):
    """
    Extract data from all subdirectories in a folder.

    Parameters:
        dir_name (str): Directory name
        V (array): Voltage grid
        n_points (int): Number of voltage points
        num_IdVg (int): Number of Id-Vg curves
        num_feats (int): Number of features
        minval (float): Minimum current value

    Returns:
        X, Y (tuple): Raw extracted data arrays
    """
    subdirs = sorted(glob.glob(dir_name + "/*"))
    counter = 0
    crit_mass = 10000
    tick = time.time()
    X_array_final = []
    Y_array_final = []
    X_array = []
    Y_array = []
    num_saves = 0

    for subdir in subdirs:
        if counter % crit_mass == 0 and counter > 1:
            num_saves += 1
            X_array_final.append(X_array)
            Y_array_final.append(Y_array)
            X_array = []
            Y_array = []
            num_saves += 1

        counter += 1

        if counter % 250 == 0:
            tock = time.time()
            print(counter, tock - tick, np.shape(X_array))
            tick = tock

        y, variable_names = build_y_array(subdir + "/variables.csv")
        x = np.array([])

        subdir_files = glob.glob(subdir + "/*")
        subdir_files = sorted(subdir_files)
        Flag = False
        Id = 0

        for filename in subdir_files:
            if not "IdVg" in filename and not "IdVd" in filename:
                continue
            try:
                x = build_x_array(x, filename, V, minval)
            except Exception as e:
                print(e)
                Flag = True
                continue

        if not x.size == (num_feats * num_IdVg * n_points) or Flag:
            continue

        x = np.array(x)
        x = np.reshape(x, (num_feats * num_IdVg, n_points))
        x = x.T
        x = np.reshape(x, (1, n_points, num_feats * num_IdVg))
        X_array.append(x)
        Y_array.append(y)

        with open(dir_path + "/variable_names.txt", "w") as file:
            for string in variable_names:
                file.write(string + "\n")

    X_array_final.append(X_array)
    Y_array_final.append(Y_array)

    X = np.concatenate(X_array_final, axis=0)
    Y = np.concatenate(Y_array_final, axis=0)

    X = X.reshape(X.shape[0], X.shape[2], X.shape[3])
    Y = Y.reshape(Y.shape[0], Y.shape[2])

    return X, Y


def build_x_array(x, filename, V, minval):
    """
    Build feature array from a single current-voltage file.

    Parameters:
        x (array): Existing feature array
        filename (str): Path to current-voltage file
        V (array): Voltage grid
        minval (float): Minimum current value

    Returns:
        x (array): Updated feature array
    """
    # sentaurus and hemt simulations are formatted differently
    if NNv.simtype == "sentaurus":
        data = np.loadtxt(filename, skiprows=1, delimiter=",").T
    elif NNv.simtype == "hemt":
        data = np.loadtxt(filename, usecols=range(2)).T
        data[1] = np.abs(data[1])

    Id = interpolate_data([data[0], data[1]], V, logscale=False)
    Id_log = interpolate_data([data[0], np.log10(np.abs(data[1]))], V, logscale=False)

    indices = np.where(Id < minval)
    Id[indices] = minval
    indices = np.where(Id_log < np.log10(minval))
    Id_log[indices] = np.log10(minval)

    Id_grad = np.gradient(Id, V)
    Id_grad_log = np.gradient(Id_log, V)

    x = np.concatenate((x, Id, Id_log, Id_grad, Id_grad_log))
    return x


def build_y_array(filename):
    """
    Build parameter array from variables CSV file.

    Parameters:
        filename (str): Path to variables CSV file

    Returns:
        y (array): Parameter values
        variable_names (list): Parameter names
    """
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter="=")
        y = [float(row[1]) for row in reader]

    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter="=")
        variable_names = [str(row[0]) for row in reader]

    y = np.array(y, dtype="float64")
    y = np.reshape(y, (1, np.size(y)))
    return y, variable_names


def load_exp(
    filename,
    sheetname,
    V,
    gateVcol="GateV",
    Idcol="DrainI",
    start=1102,
    stop=1854,
    skip=1,
    W=1,
):
    """
    Load experimental data from Excel file.

    Parameters:
        filename (str): Excel file path
        sheetname (str): Sheet name
        V (array): Voltage grid
        gateVcol (str): Gate voltage column name
        Idcol (str): Drain current column name
        start (int): Start row
        stop (int): Stop row
        skip (int): Skip factor for plotting
        W (float): Width normalization factor

    Returns:
        Id_int, Id_int_log (tuple): Interpolated linear and log current
    """
    df = pd.read_excel(filename, sheet_name=sheetname, usecols=[gateVcol, Idcol])
    Vg = np.array(df["GateV"])[start:stop]
    Id = np.array(df["DrainI"])[start:stop] / W
    Id_int = interpolate_data([Vg, Id], V)
    Id_int_log = interpolate_data([Vg, np.log10(np.abs(Id))], V)

    plt.plot(Vg[::skip], Id[::skip], color="k", marker="o", ls="None")
    plt.plot(V, Id_int, color="r", ls="--")
    plt.savefig(filename.replace(".xlsx", ".png"))
    plt.close()

    plt.plot(Vg[::skip], np.log10(Id[::skip]), color="k", marker="o", ls="None")
    plt.plot(V, Id_int_log, color="r", ls="--")
    plt.savefig(filename.replace(".xlsx", "_log.png"))
    plt.close()

    return Id_int, Id_int_log


def process_device(dev, V):
    """
    Process experimental device data.

    Parameters:
        dev (str): Device name
        V (array): Voltage grid

    Returns:
        x_input (array): Processed feature array
    """
    dev_100_filename = dir_path + "/exp_data/" + dev + "_100mVds.xlsx"
    dev_1000_filename = dir_path + "/exp_data/" + dev + "_1Vds.xlsx"
    Id_100, Id_100_log = load_exp(dev_100_filename, "{}_100mVds".format(dev), V)
    Id_1000, Id_1000_log = load_exp(dev_1000_filename, "{}_1Vds".format(dev), V)

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

    x_input = np.array(
        [
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
        ]
    )

    return x_input


def process_exp(data_exp_100, data_exp_1000, new_V, minval):
    """
    Process experimental data arrays.

    Parameters:
        data_exp_100 (tuple): 100mV experimental data (Vg, Id)
        data_exp_1000 (tuple): 1V experimental data (Vg, Id)
        new_V (array): New voltage grid
        minval (float): Minimum current value

    Returns:
        x_input (array): Processed feature array
    """
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

    x_input = np.array(
        [
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
        ]
    )

    return x_input
