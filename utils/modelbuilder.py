#! /bin/env/ python3
'''
this file is written with the intention to simplify the keras model-building
process and make it cleaner.
it remains to be seen how useful it will be.
keeping it in the repo for now.
'''

import keras.backend as K
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Dropout, LeakyReLU, Flatten, LSTM, Input,
                         Masking
from keras.callbacks import ModelCheckpoint
from keras.utils import normalize

def build_dense_layers(layer_sizes=[None], drop=0):
    '''
    given a list of integer sizes of layers, b=constructs, connects,
    and returns a tuple of the beginning and final layer in this scheme,
    since the function uses keras's functional API
    ---
        layer_sizes: a list of integer sizes of dense layers to construct.
                     each size leads to the creation of a layer
        drop: float in [0, 1], the rate of dropout to apply after each
              dense layer (default: 0)
    '''
    layers = [Dense(layer_sizes[0], activation='relu', name='dense%d'%0)]
    if drop > 0:
        dropout = Dropout(drop, name='drop%d'%(i+1))
        layers += [dropout(layers[-1])]

    for i, size in enumerate(layer_sizes[1:]):
        dense = Dense(size, activation='relu', name='dense%d'%(i+1))
        layers += [dense(layers[-1])] # functional API
        if drop > 0:
            dropout = Dropout(drop, name='drop%d'%(i+1))
            layers += [dropout(layers[-1])]

    return layers[-1], layers[1]
