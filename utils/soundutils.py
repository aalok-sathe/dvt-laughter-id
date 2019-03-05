#! /bin/env/ python3
'''
this file is intended to house util functions specific to handling soundfiles
'''

# local imports
import color
import utils

# stdlib and package imports
import yaml
import itertools
from glob import glob
from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
import numpy as np
import pomegranate as pmgt
from progressbar import progressbar
from sklearn.preprocessing import normalize
from pathlib import Path


def _convertref(value=None, sr=None, to='audio'):
    '''
    converts a timestamp to audio frame index or vice versa, according to
    specified sampling rate of audio and direction requested
    ---
        value: the entity to be converted
        sr: sampling rate of audio in Hz
        to: whether to convert to audio ('audio') or to time ('time')
            where the time is in milliseconds
    '''
    try:
        if to == 'audio':
            seconds = value * 1e-3
            frame = seconds * sr
            return int(frame)
        else:
            seconds = value / sr
            millisec = seconds * 1e3
            return int(millisec)
    except (ValueError, TypeError) as e:
        color.ERR('ERR', 'please check your arguments')
        raise


def get_data_spectro(wavdata, sr, windowlen=100):
    '''
    uses scipy.signal to extract spectrogram of wavdata, and returns (n_freq,)
    shaped arrays with values averaged over windowlen milliseconds.
    with the default options, a 1-second chunk at 48kHz will produce 214
    examples from scipy, and if you pass windowlen=100, this method will return
    10 averaged chunks of 100ms each, which would be averages over 21 examples.
    returns log10 values instead of absolute raw values
    '''
    f, t, Sxx = signal.spectrogram(wavdata, sr)
    logSxx = np.log10(1+Sxx)

    samples = []
    for i in range(0, logSxx.shape[1], windowlen):
        window = logSxx[:,i:i+windowlen]
        meanSxx = np.mean(window, axis=1)
        samples.append(meanSxx)

    return np.vstack(samples)
