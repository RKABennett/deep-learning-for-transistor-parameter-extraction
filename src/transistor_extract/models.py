import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GRU, LayerNormalization, Reshape


def dummy_fn():
    '''
    We need to define all custom objects when we load our models, so this gives
    us a dummy function to use if we choose to load the forward and/or inverse
    models in this script. There's probably a less dumb way of doing this, but
    this works for now.
    '''
    pass

def build_model_forward(num_params, num_IdVg, n_points):
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

def build_model_inverse(num_points, num_IdVg, num_feats, num_params):
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

