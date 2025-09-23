import json
import os
import numpy as np
import os
from pathlib import Path
import random
import scipy.stats
import sys

import tensorflow as tf                                                         
import transistor_extract as TE

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
repo_root = Path(__file__).resolve().parents[1]
processed_data_loc = repo_root / 'data' / 'processed'
config_path = os.path.join(repo_root, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

###############################################################################
#
# Draw our data for testing and training. Here we'll train with a subset
# of our data: 500 in our training set.
#
###############################################################################
X_train_and_dev = np.load(processed_data_loc / 'X_train_and_dev.npy')
Y_train_and_dev = np.load(processed_data_loc / 'Y_train_and_dev.npy')
X_train_and_dev, Y_train_and_dev = TE.shuffle_arrays_in_unison(
                                  X_train_and_dev, Y_train_and_dev)

size = 500
dev_size = int(size*0.15)
train_size = size - dev_size

train_start = 0
train_end = train_size
dev_start = train_size
dev_end = train_size + dev_size

X_train = X_train_and_dev[train_start:train_end:]
X_dev = X_train_and_dev[dev_start:dev_end:]
X_test = np.load(processed_data_loc / 'X_test.npy')

Y_train = Y_train_and_dev[train_start:train_end]
Y_dev = Y_train_and_dev[dev_start:dev_end]
Y_test = np.load(processed_data_loc / 'Y_test.npy')

Z_train = TE.concat_X_and_Y(X_train, Y_train)
Z_dev = TE.concat_X_and_Y(X_dev, Y_dev)

###############################################################################
#
# Prepare and re-initalize our models
#
###############################################################################
model_name_forward = 'NN_forward.keras'
model_name_inverse = 'NN_inverse.keras'
model_name_inverse_pretrain = 'NN_inverse.keras'
model_name_inverse_pretrain_final = 'NN_inverse.keras'

model_forward = TE.build_model_forward(
    cfg["data"]["num_params"],
    cfg["data"]["num_IdVg"],
    cfg["data"]["n_points"]
)

model_inverse = TE.build_model_inverse(
    cfg["data"]["n_points"],
    cfg["data"]["num_IdVg"],
    cfg["data"]["num_feats"],
    cfg["data"]["num_params"]
)

model_inverse_pretrain = TE.build_model_inverse(
    cfg["data"]["n_points"],
    cfg["data"]["num_IdVg"],
    cfg["data"]["num_feats"],
    cfg["data"]["num_params"]
)

###############################################################################
#
# forward only
#
###############################################################################
model_forward, _ = TE.train_forward_NN(
    dir_path,
    X_train[:, :, 0:cfg["data"]["num_IdVg"] * 2],
    Y_train,
    X_dev[:, :, 0:cfg["data"]["num_IdVg"] * 2],
    Y_dev,
    model_forward,
    model_name_forward,
    cfg["forward_model"]["lr0_forward"],
    cfg["forward_model"]["ar_forward"],
    cfg["forward_model"]["N_anneals_forward"],
    cfg["forward_model"]["patience_forward"],
    cfg["forward_model"]["bs_forward"]
)

###############################################################################
#
# inverse with pretraining
#
###############################################################################
N_augment = 100000
Xscaling = np.loadtxt(processed_data_loc / 'Xscaling.dat')
Yscaling = np.loadtxt(processed_data_loc / 'Yscaling.dat')

X_synth, Y_synth = TE.augment_data(
    model_forward,
    N_augment,
    np.shape(Y_train)[1],
    Xscaling,
    Yscaling,
    np.linspace(cfg["data"]["Vmin"], cfg["data"]["Vmax"], cfg["data"]["n_points"]),
    save=False
)

num_synth_train = int(0.85*N_augment)
X_train_synth = X_synth[0:num_synth_train]
Y_train_synth = Y_synth[0:num_synth_train]
X_dev_synth = X_synth[num_synth_train:]
Y_dev_synth = Y_synth[num_synth_train:]
Z_train_synth = TE.concat_X_and_Y(X_train_synth, Y_train_synth)
Z_dev_synth = TE.concat_X_and_Y(X_dev_synth, Y_dev_synth)

model_inverse_pretrain, _ = TE.train_inverse_NN(
    dir_path,
    X_train_synth,
    Z_train_synth,
    X_dev_synth,
    Z_dev_synth,
    model_inverse_pretrain,
    model_name_inverse_pretrain,
    model_forward,
    cfg["inverse_pretraining"]["lr0_inverse_pre"],
    cfg["inverse_pretraining"]["ar_inverse_pre"],
    cfg["inverse_pretraining"]["N_anneals_inverse_pre"],
    cfg["inverse_pretraining"]["patience_inverse_pre"],
    cfg["inverse_pretraining"]["bs_inverse_pre"]
)

model_inverse_pretrain_final, _ = TE.train_inverse_NN(
    dir_path,
    X_train,
    Z_train,
    X_dev,
    Z_dev,
    model_inverse_pretrain,
    model_name_inverse_pretrain_final,
    model_forward,
    cfg["inverse_finetuning"]["lr0_inverse_ft"],
    cfg["inverse_finetuning"]["ar_inverse_ft"],
    cfg["inverse_finetuning"]["N_anneals_inverse_ft"],
    cfg["inverse_finetuning"]["patience_inverse_ft"],
    cfg["inverse_finetuning"]["bs_inverse_ft"]
)
