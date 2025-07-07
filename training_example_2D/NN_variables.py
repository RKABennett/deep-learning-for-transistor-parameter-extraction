import os
import random
import sys
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Attention, BatchNormalization, 
                                     Dense, Dropout, GRU, GroupNormalization,
                                     Input, LayerNormalization, LSTM, Reshape)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import (MeanAbsolutePercentageError, 
                                     RootMeanSquaredError)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam

import NN_fns as NNf

np.random.seed(19700101)
random.seed(19700101)
tf.random.set_seed(19700101)
dir_path = os.path.dirname(os.path.abspath(__file__))

###############################################################################
#
# Settings associated with IdVg data
#
###############################################################################

data_folder = dir_path + '/data'

num_params = 8 # number of fitting parameters to extract
simtype = 'sentaurus' # 'sentaurus' or 'hemt' (verilog-a) simulations
n_points = 32   # number of points we want on each IdVg curve
num_IdVg = 2    # number of IdVg curves
num_feats = 4   # number of features we get per IdVg curve. 
V = np.linspace(-5.9, 49.9, n_points) # Voltage vector for interpolation
minval = 5e-11                   # noise floor we impose on our data

Xscaling_name = 'Xscaling.dat'
Yscaling_name = 'Yscaling.dat'

###############################################################################
#
# Forward model settings
#
###############################################################################
lr0_forward = 1e-3      # initial learning rate for the forward solve
ar_forward = 0.35       # annealing rate for the forward solve
N_anneals_forward = 4   # number of annealing steps
patience_forward = 50   # patience for early stopping
bs_forward = 128        # batch size

###############################################################################
#
# Training (without pre-training) settings
#
###############################################################################
lr0_inverse = 1e-3
ar_inverse = 0.8
N_anneals_inverse = 10
patience_inverse = 40
bs_inverse = 256

###############################################################################
#
# Data augmentation and pre-training settings
#
###############################################################################
lr0_inverse_pre = 2.5e-4
ar_inverse_pre = 0.6
N_anneals_inverse_pre = 2
patience_inverse_pre = 5
bs_inverse_pre = 256

###############################################################################
#
# Fine-tuning settings
#
###############################################################################
lr0_inverse_ft = 2e-5
ar_inverse_ft = 0.9
N_anneals_inverse_ft = 3
patience_inverse_ft = 40
bs_inverse_ft = 256

###############################################################################
#
# Neural networks
#
###############################################################################

def dummy_fn():
    '''
    We need to define all custom objects when we load our models, so this gives
    us a dummy function to use if we choose to load the forward and/or inverse
    models in this script. There's probably a less dumb way of doing this, but
    this works for now.
    '''
    pass

def build_model_forward(num_params, num_IdVg):
    '''
    Build our forward neural network, i.e., the neural network we use to 
    emulate the current-voltage model. We will use this network to build a 
    pretraining set and to evaluate our loss function during training.
    '''
    input_layer_forward = Input(shape=(num_params,))
    NN_forward = input_layer_forward
    NN_forward = Dense(1024, activation='tanh')(NN_forward)
    NN_forward = Reshape((n_points, 32))(NN_forward) 
    NN_forward = GRU(512, return_sequences=True, activation='tanh',
                     recurrent_activation='sigmoid')(NN_forward)
    NN_forward = LayerNormalization()(NN_forward)
    output_layer_forward = Dense(num_IdVg*2, activation='tanh')(NN_forward)
    model_forward = Model(inputs=input_layer_forward,
                          outputs=output_layer_forward)

    model_forward.set_weights(
                              [tf.keras.initializers.glorot_uniform()(w.shape)
                               for w in model_forward.get_weights()]
                              )

    model_forward.summary()    
    return model_forward

def build_model_inverse_smaller(num_points, num_IdVg, num_feats, num_params):
    '''
    num points: number of IdVg points
    num curves: number of IdVg curves
    num output: number of parameters
    '''
    input_layer_inverse = Input(shape=(num_points, num_IdVg*num_feats))
    NN = tf.keras.layers.Flatten()(input_layer_inverse)
    NN = Dense(2048, activation='relu')(NN)
    NN = LayerNormalization()(NN)
    NN = Dense(2048, activation='relu')(NN)
    output_layer_inverse = Dense(
                                 num_params+num_points*num_IdVg*2, 
                                 activation='tanh'
                                 )(NN)
    model_inverse = Model(
                          inputs=input_layer_inverse, 
                          outputs=output_layer_inverse
                          )
    model_inverse.summary()
    
    model_inverse.set_weights(
                              [tf.keras.initializers.glorot_uniform()(w.shape)
                               for w in model_inverse.get_weights()]
                              )
    model_inverse.summary()
    return model_inverse

def build_model_inverse_GRU():
    input_layer_inverse = Input(shape=(32, 4*num_IdVg))
    NN_inverse = GRU(
             256,
             return_sequences=True,
             activation='tanh',
             recurrent_activation='sigmoid')(input_layer_inverse)

    output_layer_inverse = GRU(
                       8,
                       return_sequences=False,
                       activation='tanh',
                       recurrent_activation='sigmoid')(NN_inverse)

    model_inverse = Model(
                          inputs=input_layer_inverse,
                          outputs=output_layer_inverse
                          )
    model_inverse.summary()
    return model_inverse

def build_model_inverse(num_points, num_curves, num_output):
    input_layer_inverse = Input(shape=(num_points, num_curves))
    NN = tf.keras.layers.Flatten()(input_layer_inverse)
    for i in range(9):
        NN = Dense(1024, activation='relu')(NN)
        NN = LayerNormalization()(NN)

    NN = Dense(1024, activation='relu')(NN)
    
    output_layer_inverse = Dense(
                                 num_output+num_points*num_curves, 
                                 activation='tanh'
                                 )(NN)
    model_inverse = Model(inputs=input_layer_inverse, 
                          outputs=output_layer_inverse
                          )

    model_inverse.set_weights(
                              [tf.keras.initializers.glorot_uniform()(w.shape)
                               for w in model_inverse.get_weights()]
                              )
    model_inverse.summary()
    return model_inverse




