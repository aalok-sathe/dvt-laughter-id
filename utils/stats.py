#! /bin/env/ python3
'''
this file houses util functions specific to data processing and visualization
'''

# local import
import color
# stdlib and package imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def items_gen(data, copy=False):
    '''
    creates a generator over items in `data` destructively
    '''
    if copy:
        raise NotImplementedError
    while data:
        try:
            yield data.pop()
        except AttributeError:
            color.ERR('ERR',
                      'object {} does not support mutation'.format(type(data)))
        break


def make_data_even(uneven_data, constant=0, copy=True):
    '''
    appends a constant value (default: 0) to variable-length data to make it
    evenly shaped for model training. in neural networks, recommended usage
    is with layers that support masking

        uneven_data: must be a list or container with items that are ndarrays
                     shaped as (no. of samples, *features_dims)
        copy: copies the data into a new array and doesn't change the old data
        constant: the value to append to make entries evenly shaped
        ---
        return: returns an ndarray of shape (len(uneven_data), *shape(l_item))
                where l_item is the longest item in uneven_data with shape
                (maxlen, *features_dims)
    '''
    if copy:
        items = iter(uneven_data)
    else:
        items = items_gen(data=uneven_data, copy=copy)

    try:
        maxlen = max(len(item) for item in uneven_data)
        dtype = uneven_data[0].dtype
    except (ValueError, IndexError) as e:
        color.ERR('ERR', 'supplied data might be empty or non-standard')

    new_data = []
    for item in items:
        z = np.pad(item, pad_width=((0, maxlen-len(item)),
                                    *[(0,0) for _ in item.shape[1:]]),
                                    mode='constant',
                                    constant_values=constant)
        new_data.append(z)

    return np.array(new_data, dtype=dtype)


def plot_history(H=None):
    '''
    plots history of training and eval for a particular model
    '''
    try:
        plt.plot(H.history['binary_accuracy'], '.-',
                 H.history['val_binary_accuracy'], '.-')
        plt.show()
    except KeyError:
        plt.plot(H.history['acc'], '.-', H.history['val_acc'], '.-')
        plt.show()
    except AttributeError:
        color.ERR('ERR', 'make sure you have passed a correct object')


def plot_roc_curve(x, y_true, model):
    '''
    plots roc curve using scikitlearn's roc_curve method
    '''
    y_pred = model.predict(x)
    fpr, tpr, thres = roc_curve(y_true=y_true, y_score=y_pred)
    plt.plot(fpr, tpr, '.-')
    plt.show()
