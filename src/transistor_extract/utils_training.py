import copy
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from .utils_misc import scale_X, unscale_vector

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(root_dir, "config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))

###############################################################################
#
# Training functions
#
###############################################################################

def train_inverse_NN(
                     X_train, 
                     Y_train, 
                     X_dev, 
                     Y_dev, 
                     model_inverse, 
                     model_name_inverse,
                     model_forward,
                     lr, 
                     ar,
                     N_anneals,
                     patience,
                     bs,
                     ):

    """
    Train an inverse neural network to predict physical parameters (e.g., 
    mobility, barrier height, etc) that will allow a current simulator 
    (e.g., TCAD or a compact model) to reproduce input current-voltage 
    characteristics.
    
    Parameters:
        X_train (numpy array) 
            --  3D array containing input training data, i.e., current-voltage 
                curves and features. Should be formatted such that the first 
                dimension corresponds to different devices, the second dimension 
                corresponds to fixed Vgs points, and the third axis corresponds 
                to different features. 
        Y_train (numpy array)
            --  2D array containing output training data, i.e., the physical
                parameters that we wish to solve for. To evaluate our loss
                function, we require that the second and third dimensions of
                X_train be concatenated with the parameters, so each entry of
                Y_train contains many more entries than just the parameters. 
                However, we discard all values except for the parameters.
        X_dev (numpy array)               
            --  Same as X_train, for the development set.
        Y_dev (numpy array)
            --  Same as Y_train, for the development set
        model_inverse (tensorflow keras model)
            --  The initialized inverse model that we wish to train.
        model_name_inverse (string) 
            --  The name we wish to save our model as.
        model_forward (tensorflow keras model)
            --  Pre-trained forward neural network we use to evaluate our loss
                function.
        lr (float)
            --  Initial learning rate for training.
        ar (float)
            --  Annealing rate for learning rate annealing.
        N_anneals (int)
            --  Number of annealing cycles. (Use N_anneals = 1 to avoid
                annealing.)
        patience (int)
            --  Patience used for early stopping
        bs (int)
            -- Mini-batch size
    
    Returns:
        The trained inverse network and the val loss history during training.
    """

    # We need the forward model to be accessible in our forward loss function
    # so here we redefine it as a global variable. This probably isn't the
    # cleanest approach, but it works for now.
    global model_forward_pretrained
    model_forward_pretrained = model_forward
    
    val_loss_history = []
    cp = ModelCheckpoint(
                         dir_path + '/' + model_name_inverse, 
                         save_best_only=True
                         )
    es = EarlyStopping(
                       monitor='val_loss', 
                       patience=patience, 
                       restore_best_weights=True
                       )

    # Check the val_loss before we begin training and save the starting
    # weights. If training does not improve our network, we revert back to 
    # these at the end.
    model_inverse.compile(
        loss=surrogate_loss,
        optimizer=Adam(learning_rate=lr),
        jit_compile=False
        )
    starting_val_loss_original = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
    pretrained_weights_original = model_inverse.get_weights()

    # Learning rate annealing loop
    for i in range(N_anneals):
        
        # Check the val_loss before each training loop and save the starting
        # weights. If the loop does not improve our network, we revert back to 
        # these at the end.
        starting_val_loss = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
        starting_weights = model_inverse.get_weights()

        # Setting jit_compile=False is necessary to avoid an error when using
        # a GRU-based forward neural network to evaluate our loss function
        # for a dense inverse NN. This could be system dependent; if you
        # encounter a compilation error during training, removing jit_compile
        # below could be a good starting point.
        model_inverse.compile(
                              loss=surrogate_loss, 
                              optimizer=Adam(learning_rate=lr), 
                              jit_compile=False 
                              )

        # We set a huge number of epochs because we train with early stopping.
        model_fit = model_inverse.fit(
                                      X_train, 
                                      Y_train, 
                                      validation_data=(X_dev, Y_dev),
                                      epochs=10**10,
                                      callbacks=[cp, es], 
                                      batch_size=bs
                                      )

        val_loss_history.extend(model_fit.history['val_loss'])
        lr *= ar


        # Check the val loss after the training loop and compare it to before
        # the training loop. If the val loss has gotten worse, we return to
        # the weights before the training loop.
        current_val_loss = model_inverse.evaluate(X_dev, Y_dev, verbose=0)
        current_weights = model_inverse.get_weights()
        print("Starting val loss = {}, current val_loss = {}".format(
                                                        starting_val_loss, 
                                                        np.min(val_loss_history)
                                                        )
                                                        )
        
        if starting_val_loss < current_val_loss:
            print("Resetting weights for this cycle")
            model_inverse.set_weights(starting_weights)
        else:
            print("Updating weights for this cycle")
            model_inverse.load_weights(dir_path + '/' + model_name_inverse)
    
    # Display helpful information after training is complete.
    print("TRAINING COMPLETE.".format(
                                      starting_val_loss, 
                                      np.min(val_loss_history)
                                      )
                                      )

    print("Pretrain val loss = {}, current val_loss = {}".format(
                                                       starting_val_loss, 
                                                       np.min(val_loss_history)
                                                       )
                                                       )

    # Reset our weights if the full training cycle did not improve our val loss
    if starting_val_loss_original < np.min(val_loss_history):
        print("Resetting weights")
        model_inverse.set_weights(pretrained_weights_original)
    else:
        print("Updating weights")
        model_inverse.load_weights(dir_path + '/' + model_name_inverse)
    return model_inverse, val_loss_history

def train_forward_NN(
                     Id_train,
                     params_train,
                     Id_dev,
                     params_dev,
                     model_forward,
                     model_name_forward,
                     lr,
                     ar,
                     N_anneals,
                     patience,
                     bs
                     ):

    """
    Train a forward neural network to predict current-voltage characteristics
    based on input parameters such as mobility and Schottky barrier height. 
    This network mimics a physics-based TCAD model or a compact model; we use
    it to generate a pre-training set and to evaluate the loss function of
    our inverse neural network.
    
    Parameters:
        Id_train (numpy array) 
            --  3D array containing OUTPUT training data, i.e., current-voltage
                characteristics. The first dimension corresponds to the device;
                the second to the fixed Vgs grid, and the third to the current
                itself. Here, along that third axis, we consider the linear-
                and log10 of Id at each Vds considered, giving a total of
                2*(number of Vds measured) features.

        params_train (numpy array)
            --  2D array containing the INPUT training data, i.e., the physical
                parameters that our current-voltage model accepts, e.g., 
                mobility, Schottky barrier height.
        Id_dev (numpy array)               
            --  Same as Id_train, for the development set.
        params_dev (numpy array)
            --  Same as params_train, for the development set
        model_forward (tensorflow keras model)
            --  The initialized forward model that we wish to train.
        model_name_forward (string) 
            --  The name we wish to save our model as.
        model_forward (tensorflow keras model)
            --  Pre-trained forward neural network we use to evaluate our loss
                function.
        lr (float)
            --  Initial learning rate for training.
        ar (float)
            --  Annealing rate for learning rate annealing.
        N_anneals (int)
            --  Number of annealing cycles. (Use N_anneals = 1 to avoid
                annealing.)
        patience (int)
            --  Patience used for early stopping
        bs (int)
            -- Mini-batch size
    
    Returns:
        The trained forward network and the val loss history during training.
    """

    val_loss_history = []
    cp = ModelCheckpoint(
                         dir_path + '/' + model_name_forward,
                         save_best_only=True
                         )
    es = EarlyStopping(
                       monitor='val_loss',
                       patience=patience,
                       restore_best_weights=True
                       )

    model_forward.compile(
                  loss=CombinedMSELoss(),
                  optimizer=Adam(learning_rate=lr),
                  metrics = [RootMeanSquaredError(),
                             MeanAbsolutePercentageError()]
                  )

    # Learning rate annealing loop
    for i in range(N_anneals):
        # Check the val_loss before each training loop and save the starting
        # weights. If the loop does not improve our network, we revert back to 
        # these at the end.
        starting_val_loss = model_forward.evaluate(
                                                   params_dev, 
                                                   Id_dev, 
                                                   verbose=0)[0]
        starting_weights = model_forward.get_weights()

        model_forward.compile(
                      loss=CombinedMSELoss(),
                      optimizer=Adam(learning_rate=lr),
                      metrics = [RootMeanSquaredError(),
                                 MeanAbsolutePercentageError()]
                      )
        # We set a huge number of epochs because we train with early stopping.
        model_fit = model_forward.fit(
                              params_train,
                              Id_train,
                              validation_data=(params_dev,Id_dev),
                              epochs=10**10,
                              callbacks=[cp,es],
                              batch_size = bs
                              )
        val_loss_fn = np.min(model_fit.history['val_loss'])
        val_loss_history.append(val_loss_fn)
        lr = lr*ar

        # Check the val loss after the training loop and compare it to before
        # the training loop. If the val loss has gotten worse, we return to
        # the weights before the training loop.
        current_val_loss = model_forward.evaluate(
                                                  params_dev, 
                                                  Id_dev, 
                                                  verbose=0
                                                  )[0]
        current_weights = model_forward.get_weights()
        print("Start val loss = {}, current val_loss = {}".format(
                                                        starting_val_loss, 
                                                        np.min(val_loss_history)
                                                        )
                                                        )

        if starting_val_loss < current_val_loss:
            print("Resetting weights for this cycle")
            model_forward.set_weights(starting_weights)
        else:
            print("Updating weights for this cycle")
            model_forward.load_weights(dir_path + '/' + model_name_forward)

    return model_forward, val_loss_history

###############################################################################
#
# Custom loss functions
#
###############################################################################

def surrogate_loss(Y_true, Y_pred):
    '''
    Custom loss function for our inverse neural network. Our loss here is:

        sqrt(MSE*L_Id)

    where 

        MSE is the standard mean square error for our model parameters.

        L_Id is a term that describes how much error we have in our
        original and predicted current based on the parameters that we are
        estimating.

    Here, we evaluate L_Id by first computing the true and predicted currents:

        Id_true = measured current
        Id_predicted = f(Y_predicted)
        where f is a pre-trained forward neural network; it must be a globally
        defined variable named 'model forward pretrained' so that we can 
        access it here.

    and then L_Id is calculated based on the difference of the actual vs. 
    predicted current, and its 1st and 2nd derivatives, in both linear and 
    log space.

    Note that Id_true is our input to the inverse neural network. There is no
    direct way to access the network's inputs while evaluating the loss fn;
    thus, before training begins, we concatenate the MOSFET physical parameters
    with the current into a combined vector, which we feed into the NN as our
    intended output vector. We use the values of Id in the output vector only
    when evaluating Lid and discard them for the rest of the loss fn.
    '''

    if 'model_forward_pretrained' not in globals():
        raise NameError('''The forward model needs to be a global variable
                         named \'model_forward_pretrained\' ''')

    # take only the few relevant parameters at the start of the Y vectors to
    # evaluate the MSE of the parameter error and the current values to
    # evaluate L_id
    mse_Y = MeanSquaredError()(
    Y_true[:, 0:cfg["data"]["num_params"]],
    Y_pred[:, 0:cfg["data"]["num_params"]]
    )

    Id_true = Y_true[:, 8:] 

    Id_pred = model_forward_pretrained(Y_pred[:, 0:cfg["data"]["num_params"]])
    Id_pred = tf.transpose(Id_pred, perm=[0, 2, 1])
    Id_pred = tf.reshape(Id_pred, [-1, 2 * cfg["data"]["num_IdVg"] * cfg["data"]["n_points"]])

    mse_Id = MeanSquaredError()(Id_true, Id_pred)

    Id_true_1st_deriv = Id_true[:, 1:] - Id_true[:, :-1]
    Id_pred_1st_deriv = Id_pred[:, 1:] - Id_pred[:, :-1]
    mse_1st_deriv = MeanSquaredError()(Id_true_1st_deriv, Id_pred_1st_deriv)

    Id_true_2nd_deriv = Id_true_1st_deriv[:, 1:] - Id_true_1st_deriv[:, :-1]
    Id_pred_2nd_deriv = Id_pred_1st_deriv[:, 1:] - Id_pred_1st_deriv[:, :-1]
    mse_2nd_deriv = MeanSquaredError()(Id_true_2nd_deriv, Id_pred_2nd_deriv)

    total_loss = mse_Y**0.5*(mse_Id + mse_1st_deriv + mse_2nd_deriv)**0.5

    return(total_loss)


class CombinedMSELoss(tf.keras.losses.Loss):

    '''
    Custom loss function for our inverse neural network. Our loss here is:

        MSE(Id) + MSE(delta (Id)) + MSE(delta (delta (Id)))
    
    i.e., similar to MSE(Id) + MSEs of its first and second derivatives.

    '''
    def call(self, Y_true, Y_pred):
        mse_loss = MeanSquaredError()

        mse_Id = mse_loss(Y_true, Y_pred)

        delta_Y_true = (Y_true[:, 1:] - Y_true[:, :-1]) 
        delta_Y_pred = (Y_pred[:, 1:] - Y_pred[:, :-1]) 
        mse_deltaId = mse_loss(delta_Y_true, delta_Y_pred)
        
        deltadelta_Y_true = (delta_Y_true[:, 1:] - delta_Y_true[:, :-1])
        deltadelta_Y_pred = (delta_Y_pred[:, 1:] - delta_Y_pred[:, :-1])
        mse_deltadeltaId = mse_loss(deltadelta_Y_true, deltadelta_Y_pred)

        total_loss = mse_Id + mse_deltaId + mse_deltadeltaId
        return total_loss

###############################################################################
#
# Data augmenting
#
###############################################################################

def augment_data(
                 model_forward, 
                 N_augment, 
                 N_features,
                 Xscaling, 
                 Yscaling,
                 V,
                 save = True
                 ):
    
    """
    Generate augmented training data using our forward neural network.

    Parameters:
        model_forward (tensorflow keras model)
            --  The trained forward neural network used to generate data.
        N_augment (int)
            --  The number of devices in the desired dataset.
        Xscaling_name (str)
            --  Filepath + name for the X array (current-voltage) scaling 
                parameters used when generating the original dataset.
        Xscaling_name (str)
            --  Same as above, for the Y array.
        V (array-like)
            --  Array or list corresponding to the fixed Vgs points we used.
    
    Returns:
        Augmented data in X and Y arrays.
    """
    # generate random input parameters and then call forward model to predict
    # the current-voltage characteristics
    Y = np.random.uniform(-1, 1, (N_augment, N_features))
    print(np.shape(Y))
    currents_generated = np.array(model_forward.predict(Y))

    # load scaling parameters
    Xmins = Xscaling[0,:]
    Xmaxs = Xscaling[1,:]
    Ymins = Yscaling[0,:]
    Ymaxs = Yscaling[1,:]

    # unscale the currents
    currents_unscaled = copy.deepcopy(currents_generated)
    for i in range(cfg["data"]["num_IdVg"] * 2):
        Xmin = Xmins[i]
        Xmax = Xmaxs[i]
        currents_unscaled[:,:,i] = unscale_vector(
                                                  currents_generated[:,:,i], 
                                                  Xmin, 
                                                  Xmax
                                                  )

    # build a new X array from the unscaled currents
    X = []
    Id     = currents_unscaled[:, :, ::2]
    Id_log = currents_unscaled[:, :, 1::2]

    Id_grad     = np.gradient(Id, V, axis=1, edge_order=2)
    Id_log_grad = np.gradient(Id_log, V, axis=1, edge_order=2)

    X_array = np.empty((N_augment, cfg["data"]["num_IdVg"] * 4, cfg["data"]["n_points"]))
    for j in range(cfg["data"]["num_IdVg"]):
        X_array[:, j*4,     :] = Id[:, :, j]
        X_array[:, j*4 + 1, :] = Id_log[:, :, j]
        X_array[:, j*4 + 2, :] = Id_grad[:, :, j]
        X_array[:, j*4 + 3, :] = Id_log_grad[:, :, j]

# Work our X array into the correct formatting
    X_array = X_array.transpose(0, 2, 1)
    X = [x_input[np.newaxis, ...] for x_input in X_array]
    X = np.array(X)
    X = np.reshape(X, (N_augment, cfg["data"]["n_points"], 4 * cfg["data"]["num_IdVg"]))

    # Right now, the X array isn't sorted properly: we want all of the currents
    # and then all of the derivatives, but it alternates between currents and
    # derivatives right now. We fix this here.


    current_indices = []
    deriv_indices = []
    for i in range(cfg["data"]["num_IdVg"]):
        current_indices.append(0 + i*4)
        current_indices.append(1 + i*4)
        deriv_indices.append(2 + i*4)
        deriv_indices.append(3 + i*4)

    X = np.concatenate([
        X[:, :, current_indices],
        X[:, :, deriv_indices]
    ], axis=-1)

    # Finally, scale the new data.
    X, Xmins_new, Xmaxs_new = scale_X(
        X,
        minarrs=Xmins[0:cfg["data"]["num_IdVg"] * 4],
        maxarrs=Xmaxs[0:cfg["data"]["num_IdVg"] * 4]
    )

    return X, Y



