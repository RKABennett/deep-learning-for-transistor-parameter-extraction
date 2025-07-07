###############################################################################
#
# Import functions
#
###############################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/rob/2D_ML/data_for_paper')))

import matplotlib as mpl
import sys
import glob as glob
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd                                                             
import random
import scipy.stats
from sklearn.utils import shuffle
import time

import tensorflow as tf                                                         
from tensorflow.keras.layers import Input, GRU, LayerNormalization, BatchNormalization, GroupNormalization, Dense, Attention, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model                                  
from tensorflow.keras.layers  import *                                          
from tensorflow.keras.callbacks import ModelCheckpoint                          
from tensorflow.keras.losses import MeanSquaredError                            
from tensorflow.keras.metrics import RootMeanSquaredError                       
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import NN_fns as NNf
import NN_variables as NNv

# seed for reproducibility 
tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

# working directory
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path + '/../')

###############################################################################
#
# Load data from saved files in the current working directory
#
###############################################################################

X_test = np.load(dir_path + '/X_test.npy')
Y_test = np.load(dir_path + '/Y_test.npy')
quantile = 0.05

#######################################################################
#
# Prepare and re-initalize our models
#
#######################################################################

model_name_inverse = 'NN_inverse.keras'

error_filename_inverse = 'errors_inverse.dat'


model_inverse = load_model(                                             
                model_name_inverse,                                      
                custom_objects={'surrogate_loss': NNv.dummy_fn}             
                ) 

model_forward_fully_trained = load_model(                                             
                'NN_forward_fully_trained.keras',                                      
                custom_objects={'CombinedMSELoss': NNv.dummy_fn}             
                ) 

_, _, errors = NNf.test_model_inverse_current(
    X_test,
    Y_test,
    model_forward_fully_trained,
    model_inverse,
    NNv.Xscaling_name,
    NNv.Yscaling_name,
    NNf.calc_R2,
    error_filename_inverse,
    deriv_error = False,
    plot = True,
    save_fits = True,
    fit_name = model_name_inverse.replace('.keras', '')
    )


