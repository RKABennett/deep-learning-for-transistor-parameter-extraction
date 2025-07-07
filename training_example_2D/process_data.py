###############################################################################
#
# Import and define functions
#
###############################################################################

import os
import sys

import NN_fns as NNf
import NN_variables as NNv
import traceback
import glob
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd                                                             
import random
import time
from scipy.interpolate import interp1d
import csv

# seed for reproducibility
np.random.seed(19700101)
dir_path = os.path.dirname(os.path.abspath(__file__))

###############################################################################
#
# Initialize
#
###############################################################################

subdirs = sorted(glob.glob(NNv.data_folder + '/*'))
counter = 0
tick = time.time()
start = tick

X_array = [] # we're going to populate this with the IdVg data and derivatives
Y_array = [] # we're going to populate this with the relevant features

X, Y = NNf.process_folder(
                         dir_path + '/data', 
                         NNv.V, 
                         NNv.n_points, 
                         NNv.num_IdVg, 
                         NNv.num_feats, 
                         NNv.minval, 
                         )

###############################################################################
#
# Divide and save arrays
#
###############################################################################


N_test = 1000
N = np.shape(X)[0] - N_test

X_test = X[N:]
X_train_and_dev = X[0:N]
Y_test = Y[N:]
Y_train_and_dev = Y[0:N]

print('Final shapes:')
print('X_train_and_dev = {}; Y_train_and_dev: {}'.format(
                          np.shape(X_train_and_dev), 
                          np.shape(Y_train_and_dev)
                          )
      )

print('X_test = {}; Y_test: {}'.format(
                          np.shape(X_test), 
                          np.shape(Y_test)
                          )
      )

np.save(dir_path + '/X_train_and_dev.npy', X_train_and_dev)
np.save(dir_path + '/X_test.npy', X_test)

np.save(dir_path + '/Y_train_and_dev.npy', Y_train_and_dev)
np.save(dir_path + '/Y_test.npy', Y_test)

print(Y_test[0, :])
print(Y_test[-1, :])
