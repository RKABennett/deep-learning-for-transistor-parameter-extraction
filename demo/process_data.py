import json
import os
from pathlib import Path
import sys

import numpy as np
import transistor_extract as TE

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(root_dir, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)


repo_root = Path(__file__).resolve().parents[1]
raw_data_loc = str(repo_root / 'data' / 'raw')
processed_data_loc = str(repo_root / 'data' / 'processed')
os.makedirs(processed_data_loc, exist_ok=True)

np.random.seed(19700101)

###############################################################################
#                     
# Initialize
#                     
###############################################################################

counter = 0
X_array = [] # we're going to populate this with the IdVg data and derivatives
Y_array = [] # we're going to populate this with the relevant features

V = np.linspace(
    cfg["data"]["Vmin"],
    cfg["data"]["Vmax"],
    cfg["data"]["n_points"],
    )

X, Y = TE.process_folder(
                         raw_data_loc,
                         processed_data_loc,
                         V,
                         cfg["data"]["n_points"],
                         cfg["data"]["num_IdVg"],
                         cfg["data"]["num_feats"],
                         cfg["data"]["minval"],
                         )

###############################################################################
#                     
# Divide and save arrays
#                     
###############################################################################

N = np.shape(X)[0] - cfg["data"]["N_test"]

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

np.save(processed_data_loc + '/X_train_and_dev.npy', X_train_and_dev)
np.save(processed_data_loc + '/X_test.npy', X_test)
np.save(processed_data_loc + '/Y_train_and_dev.npy', Y_train_and_dev)
np.save(processed_data_loc + '/Y_test.npy', Y_test)

