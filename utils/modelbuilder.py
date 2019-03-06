#! /bin/env/ python3
'''
this file is written with the intention to simplify the keras model-building
process and make it cleaner for our specific task
'''
# local imports
import color
# library imports
import keras.backend as K
from keras.models import Sequential, Model, model_from_yaml
from keras.layers import Dense, Dropout, LeakyReLU, Flatten, LSTM, Input, \
                         Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.utils import normalize


def _compile_binary(model, params=dict(optimizer='rmsprop')):
    '''
    standalone method to just compile a binary classification model
    helps with brevity in other methods
    ---
        model: instance of Keras BaseModel (or Model)
    '''
    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_conv_model(*args, **params):
    '''
    builds a standard laughter ID model with either the supplied, or otherwise
    predetermined optimum parameters.
    possible parameters:
    '''
    defaults = dict(inp=129, conv0=3, conv0_kernel=16, dense0=16, drop0=.5,
                    act='relu', drop1=.5, optimizer='rmsprop')
    for key in defaults.keys():
        if key not in params:
            params[key] = defaults[key]

    if 0 < len(args) < len(defaults)-1:
        color.ERR('ERR', 'some positional arguments supplied but not enough '
                         'arguments supplied (needed: %d)' % len(defaults))
        raise ValueError
    elif len(args) == len(defaults)-1:
        keys = ['dense0', 'drop0', 'act', 'optimizer']
        for i, thing in enumerate(args):
            params[keys[i]] = thing

    # in
    inp = Input(shape=(params['inp'], 1), name='in0')

    # stack 0a
    layer = Conv1D(params['conv0'], params['conv0_kernel'],
                   activation=params['act'], name='conv0')(inp)
    layer = MaxPooling1D(params['conv0_kernel'], name='max0')(layer)
    layer = Flatten(name='flat0')(layer)
    layer = Dropout(params['drop0'], name='dr0')(layer)

    # stack 0b
    layer = Dense(params['dense0'], activation=params['act'], name='d0')(layer)
    layer = Dropout(params['drop1'], name='dr1')(layer)

    # out
    layer = Dense(1, activation='sigmoid', name='out')(layer)

    model = Model(inputs=[inp], outputs=[layer])
    model = _compile_binary(model, params)

    if params.get('verbose', True):
        model.summary()

    return model


def build_dense_model(*args, **params):
    '''
    builds a standard laughter ID model with either the supplied, or otherwise
    predetermined optimum parameters.
    possible parameters:
    '''
    defaults = dict(dense0=14, drop0=.5, dense1=9, drop1=.2, act='relu',
                    optimizer='rmsprop')
    for key in defaults.keys():
        if key not in params:
            params[key] = defaults[key]

    if 0 < len(args) < len(defaults)-1:
        color.ERR('ERR', 'some positional arguments supplied but not enough '
                         'arguments supplied (needed: %d)' % len(defaults))
        raise ValueError
    elif len(args) == len(defaults)-1:
        keys = ['dense0', 'drop0', 'dense1', 'drop1', 'act', 'optimizer']
        for i, thing in enumerate(args):
            params[keys[i]] = thing

    # in
    inp = Input(shape=(128,), name='in0')

    # stack 0
    layer = Dense(params['dense0'], activation=params['act'], name='d0')(inp)
    layer = Dropout(params['drop0'], name='dr0')(layer)

    # stack 1
    layer = Dense(params['dense1'], activation=params['act'], name='d1')(layer)
    layer = Dropout(params['drop1'], name='dr1')(layer)

    # out
    layer = Dense(1, activation='sigmoid', name='out')(layer)

    model = Model(inputs=[inp], outputs=[layer])
    model = _compile_binary(model, params)

    if params.get('verbose', True):
        model.summary()

    return model


def build_laugh_model(*args, **params):
    '''
    '''
    color.INFO('WARNING', 'this function is left here for compatibility. '
                          'please use `build_dense_model` in your methods. '
                          'this function will be deprecated in the future.')
    return build_dense_model(*args, **params)
    # raise DeprecationWarning


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
