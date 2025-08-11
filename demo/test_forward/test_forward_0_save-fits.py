import json
import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import transistor_extract as TE  
from pathlib import Path

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
repo_root = Path(__file__).resolve().parents[2]
processed_data_loc = str(repo_root / 'data' / 'processed')
config_path = os.path.join(repo_root, "config.json")

with open(config_path, "r") as f:
    cfg = json.load(f)

tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

# model from command line
parser = argparse.ArgumentParser(description="Evaluate a trained forward model.")
parser.add_argument("--model_for", required=True, help="Path to the trained forward model (.keras/.h5)")
args = parser.parse_args()

###############################################################################
#
# Load data from saved files in the current working directory
#
###############################################################################

X_test = np.load(processed_data_loc + '/X_test.npy')[:, :, 0:cfg["data"]["num_IdVg"] * 2]
Y_test = np.load(processed_data_loc + '/Y_test.npy')

###############################################################################
#
# Prepare and re-initalize our models
#
###############################################################################

model_name_forward = args.model_for
error_filename_forward = 'errors_forward.dat'

model_forward = load_model(
    model_name_forward,
    custom_objects={
        "CombinedMSELoss": TE.CombinedMSELoss,
    },
)

Xscaling = np.loadtxt(processed_data_loc + '/Xscaling.dat')
Yscaling = np.loadtxt(processed_data_loc + '/Yscaling.dat')

# make our results folder if it doesn't exist
os.makedirs(dir_path + '/forward_results', exist_ok=True)

TE.test_model_forward(
    X_test,
    Y_test,
    model_forward,
    Xscaling,
    Yscaling,
    TE.calc_R2,
    error_filename_forward,
    dir_path + '/forward_results',
    plot=True,
    fit_name=os.path.basename(model_name_forward).replace('.keras', ''),
)

