###############################################################################
#
# Import functions
#
###############################################################################

import argparse
import numpy as np
import os
import sys
import random

import time
import tensorflow as tf                                                         
from tensorflow.keras.models import load_model
import transistor_extract as TE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/rob/2D_ML/data_for_paper')))
dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))

# seed for reproducibility 
tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

# model from command line
parser = argparse.ArgumentParser(description="Evaluate a trained inverse model.")
parser.add_argument("--model_inv", required=True, help="Path to the trained inverse model (.keras/.h5)")
parser.add_argument("--model_for", required=True, help="Path to the trained forward model (.keras/.h5)")
args = parser.parse_args()

model_name_inverse = args.model_inv
model_name_forward = args.model_for


#####################################
from pathlib import Path

import json
import os

# Get the absolute path to the project root
repo_root = Path(__file__).resolve().parents[2]
processed_data_loc = str(repo_root / 'data' / 'processed')

# Build the path to config.json
config_path = os.path.join(repo_root, "config.json")

# Load the JSON config
with open(config_path, "r") as f:
    cfg = json.load(f)

##############################################################



###############################################################################
#
# Load data from saved files in the current working directory
#
###############################################################################

X_test = np.load(processed_data_loc + '/X_test.npy')
Y_test = np.load(processed_data_loc + '/Y_test.npy')
quantile = 0.05

#######################################################################
#
# Prepare and re-initalize our models
#
#######################################################################


error_filename_inverse = 'errors_inverse.dat'


model_inverse = load_model(                                             
                model_name_inverse,                                      
                custom_objects={'surrogate_loss': TE.CombinedMSELoss}             
                ) 

model_forward_fully_trained = load_model(                                             
                model_name_forward,                                      
                custom_objects={'CombinedMSELoss': TE.CombinedMSELoss}             
                ) 

Xscaling = np.loadtxt(processed_data_loc + '/Xscaling.dat')
Yscaling = np.loadtxt(processed_data_loc + '/Yscaling.dat')

# make our results folder if it doesn't exist
os.makedirs(dir_path + '/inverse_results', exist_ok=True)

_, _, errors = TE.test_model_inverse_current(
    X_test,
    Y_test,
    model_forward_fully_trained,
    model_inverse,
    Xscaling,
    Yscaling,
    TE.calc_R2,
    error_filename_inverse,
    dir_path + '/inverse_results',
    deriv_error = False,
    plot = True,
    save_fits = True,
    fit_name = model_name_inverse.replace('.keras', '')
    )


