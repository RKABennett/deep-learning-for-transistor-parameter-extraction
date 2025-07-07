###############################################################################
#
# Import functions
#
###############################################################################



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

X_test = np.load(dir_path + '/X_test.npy')[:, :, 0:NNv.num_IdVg*2]
Y_test = np.load(dir_path + '/Y_test.npy')

###############################################################################
#
# Prepare and re-initalize our models
#
###############################################################################

model_name_forward = 'NN_forward.keras'
error_filename_forward = 'errors_forward.dat'

model_forward = load_model(                                             
                model_name_forward,                                      
                custom_objects={'CombinedMSELoss': NNv.dummy_fn}             
                ) 

NNf.test_model_forward(
    X_test,
    Y_test,
    model_forward,
    NNv.Xscaling_name,
    NNv.Yscaling_name,
    NNf.calc_R2,
    error_filename_forward,
    plot = True,
    fit_name = model_name_forward.replace('.keras', '')
    )
