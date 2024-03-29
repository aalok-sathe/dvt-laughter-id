#! /bin/env/ python3
'''
this file is intended to house util functions specific to handling an
individual episode.
for some episode 'episode_i' to be processed, utils in this file expect the
existence of:
    1. the video file "../video/${episode_i}.mp4"
    2. the audio track "../wav/${episode_i}.wav"
    3. annotations data "../data/${episode_i}_${task}.yml"
'''

# local imports
import color
import utils
from vggishutils import get_embed
import soundutils
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


def load_annotations(episode, task=None):
    '''
    loads the annotations file corresponding to a particular episode.
    expects the existence of '../data/episode_i_task.yml'
    the annotations file contains start and end times in milliseconds,
    and also the attribute of each entry. this function will extract each
    such entry, and keep track of all attributes spotted across tasks for
    a particular episode.
    '''
    patches = defaultdict(list)

    if task is not None:
        pattern = '../data/{}_{}.yml'.format(episode, task)
    else:
        pattern = '../data/{}_*.yml'.format(episode)

    if not glob(pattern):
        raise ValueError('no matching file found:', episode)

    for filename in glob(pattern):
        with open(filename, 'r') as file:
            patches_ = yaml.load(file)
            for start, end, attr in patches_:
                patches[attr].append((start, end))

    return patches


def get_patches(ep, task='laughter'):
    '''
    returns patches based on annotation file of the form positive and negative
    '''
    patches = load_annotations(ep)
    laughs = patches[task]
    nolaughs = [(e1, s2) for (s1, e1), (s2, e2) in zip(laughs, laughs[1:])
                if s2-e1 > 1e3]
    return laughs, nolaughs


def _get_y_true_label(time, annot=None):
    '''
    given a timestamp 'time' and an annotations list, determines the y_true
    score for 'time' according to annotations. will return a value [0., 1.]
    depending on how much of the window (time, time+.96) has an overlap with
    the annotated patch. it is assumed that no annotation window would be
    smaller than .96s
    '''
    for s, e in annot:
        # if either endpoint of chunk lies in a patch
        if s <= time < e or s <= (time+.96e3) < e:
            # if chunk is entirely in some patch
            if s <= time < e and s <= (time+.96e3) < e:
                return 1.
            # if the end of a chunk is in a patch
            elif time < s and s <= time+.96e3 < e:
                return (time+.96e3-s) / (.96e3)
            # if the start of a chunk is in a patch
            else:
                return (e-time) / (.96e3)
    else: # didn't find any patch the chunk could be in
        return 0.


# TODO
def get_y_true_labels(episode, precision=2):
    '''
    given an episode name in the standard naming scheme, gets a list of y_true
    scores for a particular precision value
    '''
    raise NotImplementedError
    annot = load_annotations(episode)


def _convertref(value=None, sr=None, to='audio'):
    '''
    COMPATIBILITY: this method is kept here for compatibility; but will be
                   deprecated in the future. please refer to soundutils.py
    ---
    converts a timestamp to audio frame index or vice versa, according to
    specified sampling rate of audio and direction requested
    ---
        value: the entity to be converted
        sr: sampling rate of audio in Hz
        to: whether to convert to audio ('audio') or to time ('time')
            where the time is in milliseconds
    '''
    # color.INFO('WARNING', _convertref.__doc__)
    return soundutils._convertref(value, sr, to)


def _get_data_vggish(which_episodes=None, preserve_length=False,
                     archive='../data/archive', task='laughter'):
    '''
    gets embeddings data for a list of episodes.
    as a list, expects basenames of the episodes without any attribute or
    filename extension.
    returns class data generated using patches loaded for the same episodes
    ---
        which_episodes: a list of basenames (e.g. "friends-s02-e03") of
                        episodes to process (required)
                        or the name of a single episode
        use_vggish: whether to use vggish generating data (currently no
                    alternate method is implemented, but will be in the future)
        preserve_length: whether to return data as disjoint chunks of equal
                         length, or preserve length of annotated chunks and
                         return variable-length data (defaut: False)
        archive: directory housing memoized archives of the individual episodes
                 data for a task so that they don't have to be recomputed each
                 time. default: '../data/archive'. If an empty string or None
                 or anything that evaluates to False is passed, will not
                 archive the data for this run.

        return: X, Y of shape (n_samples, n_features) when not preserve_length
                              (n_samples, maxlen, n_features) when preserving
                the number of samples in these two cases would be different
                refs, a list of where each sample is from in the original ep
    '''
    if type(which_episodes) is str:
        which_episodes = [which_episodes]

    color.INFO('INFO', 'processing episodes {}'.format(str(which_episodes)))

    X, Y, refs = [], [], []

    for ep in which_episodes:
        color.INFO('INFO', 'processing {}'.format(ep))

        laughs, nolaughs = get_patches(ep, task)

        existsflag = False
        archivepath = Path(archive)
        archivepath = archivepath.joinpath(ep + '_%s_datachunks.npz' % task)

        if archive:
            # check if archives already exist
            if archivepath.exists():
                color.INFO('INFO', 'loading from {}'.format(archivepath))
                existsflag = True
                arrays = np.load(archivepath)
                this_X = arrays['X'].tolist()
                this_Y = arrays['Y'].tolist()
                this_refs = arrays['refs'].tolist()
            else:
                this_X, this_Y, this_refs = [], [], []
                sr, wavdata = wavfile.read('../wav/{}.wav'.format(ep))
        else:
            sr, wavdata = wavfile.read('../wav/{}.wav'.format(ep))

        color.INFO('INFO', 'processing %s data in %s' % (task, ep))
        if not existsflag:
            for start, end in progressbar(laughs, redirect_stdout=False):
                if existsflag: break
                if start == end: continue
                start_f, end_f = _convertref(start, sr), _convertref(end, sr)
                # print(start_f, end_f)
                try:
                    this_x,
                    utils.sess = get_embed(input_wav=wavdata[start_f:end_f],
                                           sr=sr, sess=utils.sess)
                    if preserve_length:
                        this_X += [this_x]
                        this_Y += [1]
                        this_refs += [(ep, start, end)]
                    else:
                        this_X += [x.reshape(1, -1) for x in this_x]
                        this_Y += [1 for _ in this_x]
                        this_refs += [(ep, start, end) for _ in this_x]
                # except (tf.errors.InvalidArgumentError, Exception) as e:
                except Exception as e:
                    color.ERR('INFO', 'encountered {}; resuming...\r'.format(e))
                    pass

        color.INFO('INFO', 'processing no-%s data in %s' % (task, ep))
        if not existsflag:
            for start, end in progressbar(nolaughs, redirect_stdout=True):
                if start == end: continue
                start_f, end_f = _convertref(start, sr), _convertref(end, sr)
                # print(start_f, end_f)
                try:
                    this_x, utils.sess = get_embed(input_wav=\
                                                   wavdata[start_f:end_f],
                                                   sr=sr, sess=utils.sess)
                    if preserve_length:
                        this_X += [this_x]
                        this_Y += [0]
                        this_refs += [(ep, start, end)]
                    else:
                        this_X += [x.reshape(1, -1) for x in this_x]
                        this_Y += [0 for _ in this_x]
                        this_refs += [(ep, start, end) for _ in this_x]
                # except (tf.errors.InvalidArgumentError, Exception) as e:
                except Exception as e:
                    color.ERR('INFO', 'encountered {}; resuming...\r'.format(e))
                    pass

        X += this_X
        Y += this_Y
        refs += this_refs

        this_X = np.vstack(this_X)
        this_Y = np.array(this_Y, dtype=int)
        this_refs = np.array(this_refs, dtype=object)

        if archive and not existsflag:
            np.savez_compressed(archivepath, X=this_X, Y=this_Y, refs=this_refs)

        del this_X; del this_Y; del this_refs

    if preserve_length:
        return X, Y, refs
    else:
        return np.vstack(X), np.array(Y, dtype=int), np.array(refs,
                                                              dtype=object)


def _get_data_spectro(which_episodes, preserve_length=False,
                      archive='../data/archive', task='laughter',
                      windowlen=100):
    '''
    gets spectrograph data for an episode by calling
    soundutils.get_data_spectro on chunks, repeatedly
    '''
    if type(which_episodes) == str:
        which_episodes = [which_episodes]
        
    X, Y, refs = [], [], []
    for ep in which_episodes:
        laughs, nolaughs = get_patches(ep, task)
        sr, wavdata = wavfile.read('../wav/{}.wav'.format(ep))
  

        for label in {0, 1}:
            
            color.INFO('INFO', 
                       'processing data for ep={}; label={}'.format(ep, label))
            
            for start, end in progressbar([nolaughs, laughs][label], 
                                          redirect_stdout=True):
                if start+windowlen >= end: continue
                start_f, end_f = _convertref(start, sr), _convertref(end, sr)
            
                f, t, samples = soundutils.get_data_spectro(wavdata[start_f:end_f], 
                                                            sr, windowlen=windowlen)
                if preserve_length:
                    X.append(samples)
                    Y.append(label)
                    refs.append((ep, start, end))
                else:
                    X += [s for s in samples]
                    Y += [label for _ in samples]
                    refs += [(ep, start, end) for _ in samples]
    
    X = np.vstack(X)#.reshape(*X.shape, 1)
    Y = np.vstack(Y)
    return X, Y, refs


def get_data(which_episodes=None, preserve_length=False, backend='vggish',
             archive='../data/archive', task='laughter', windowlen=100):
    '''
    wrapper method for various implementations of get_data. default: VGGish.
    for a specific method, pass the backed kwarg one of {vggish, spectro}.
    the kwarg 'windowlen' has no effect with VGGish, but is passed to 
    the spectrogram based implementation
    '''
    if backend == 'vggish':
        return _get_data_vggish(which_episodes, preserve_length, archive, task)

    if backend == 'spectro':
        return _get_data_spectro(which_episodes, preserve_length, archive, task,
                                 windowlen)


def score_continuous_data(wavdata=None, sr=None, model=None, precision=3, L=1,
                          archive='../data/archive', name='.noname',
                          norm=False):
    '''
    Given wavdata of an audio signal and its sampling rate, this method
    will generate more embeddings for the same data than are typically needed
    for training, in order to get a better estimate of where in the audio
    there's canned laughter.
    ---
        wavdata: raw wavdata of an audio signal; typically, an entire episode
        sr: audio sampling rate (frames per second; Hz)
        model: the model to used to assign probability scores to an embedding.
               must be an instance of the Keras Model or BaseModel class
               and support prediction of data.
        precision: number of embeddings to generate, each with an equally
                   spaced-out offset less than 0.96s so that each embedding is
                   generated over a unique time interval. this number will also
                   determine the precision of the labeling, to the nearest
                   (.96/precision) seconds. note than generating an embedding
                   for each precision-point takes time and memory, so high
                   precision and memory-and-time constraints need to be
                   balanced for an optimal level of precision (default: 3;
                   min: 1).
        L: the length of the sequence the model accepts to make predictions
           about labels. (defaut: 1) [WIP; not implemented]. any value other
           than 1 would result in an Exception.

        return: outputs a (len(wavdata)*precision/(sr*.96-L), n_classes) shaped
                array of labels predicted by the model supplied

    '''
    offsets = np.arange(0, 0.96, 0.96/precision)

    archivepath = Path(archive)
    archivepath = archivepath.joinpath(name + '_emb_prec=%d.npz' % precision)

    if archivepath.exists():
        color.INFO('INFO', 'loading archived data from {}'.format(archivepath))
        data = np.load(archivepath)
        embs = data['embs'] #if precision > 1 else data['embs']
    else:
        color.INFO('INFO', 'no archived data found at {}'.format(archivepath))
        embs = []
        for x in offsets:
            start_f = int(sr*x)
            color.INFO('INFO',
                       'computing embedding for offset {}; '
                       'this may take a while'.format(x))

            emb, utils.sess = get_embed(input_wav=wavdata[start_f:], sr=sr,
                                        sess=utils.sess)
            if norm:
                emb = normalize(np.stack(emb, axis=0))
            embs.append(emb)
        np.savez_compressed(archivepath, embs=np.array(embs))

    color.INFO('INFO', 'unpacking offset embeddings into single list')
    sequence = [*sum(zip(*embs), ())] #if precision > 1 else np.vstack(embs)

    color.INFO('INFO', 'making predictions')
    preds = []
    for item in sequence:
        # print(item.shape)
        pred = model.predict(x=item.reshape(1, -1))
        preds.append(pred)

    return np.vstack(preds)
    # color.INFO('FUTURE', 'WIP; not yet implemented')
    # raise NotImplementedError


def _binary_probs_to_multiclass(binary_probs=None):
    '''
    Helper method to convert an array of binary probabilities to multiclass
    probabilities. This is necessary because a multiclass probabilities array
    specifies a probability for each class, whereas, a binary array
    '''
    assert binary_probs.shape[-1] == 1, 'badly shaped binary probabilities'
    color.INFO('INFO', 'converting binary probs array to multiclass')
    multi = [np.array([1-x, x]) for x in binary_probs]
    return np.array(multi).reshape(-1, 2)


def decode_sequence(probs=None, algorithm='threshold', params=dict(n=5, t=.8),
                    verbose=True):
    '''
    Once a model outputs probabilities for some sequence of data, that
    data shall be passed to this method. This method will use various
    ways to decode an underlying sequence in order to determine where
    the *actual* canned laughter was.
    possible algorithms to decode sequence:
        - 'neural'
          surround-n-gram neural network: this method will use a pretrained
          Keras model to label some sample i using the multiclass probabilities
          of all of the samples numbered [i-n, i-n+1, ... i, i+1, ..., i+n],
          i.e., n before and n afterwards.
        - 'hmm'
          HMM: this method will use a hidden Markov model with underlying
               states that are the same as surface states (the two state spaces
               for hidden and observed are equivalent).
               uses Viterbi to decode the underlying state sequence.
               requires a params to be passed as dict(c=DiscreteDistribution)
               where c is a class (label) and DiscreteDistribution is an
               instance of emission probabilities created using `pomegranate`,
               for each such class c (0, 1, 2, ...)
        - 'threshold'
          window and threshold method: this is simple heuristic-based method
          that will observe windows of length n, and if the average probability
          of any single class is at least t, it will assign that same
          class to all of the samples in that window. imagine a threshold of
          0.9, then it is intuitively likely if few of the samples are labeled
          with some other class, they may have been accidentally so-labeled.
        - 'modethreshold'
          like 'threshold' but instead of considering avg probability, it
          considers what percentage of labels are a particular class and if
          that surpasses a threshold, then all labels are made that same label
    ---
        probs: an nparray of (n_samples, n_classes) probabilities such that
               foreach sample, the sum of probabilities across classes adds up
               to 1. In case supplied array is of shape (n_samples,) it will be
               converted to multiclass using this module's
               _binary_probs_to_multiclass method

        return: a list of len n_samples, with the ith sample being the
                predicted label of that sample. this prediction would usually
                also incorporate somehow the samples before and after the
                current sample
    '''
    color.INFO('INFO', 'shape of input probs is: {}'.format(probs.shape))
    if probs.shape[-1] == 1:
        probs = _binary_probs_to_multiclass(probs)

    color.INFO('INFO', 'received probs of shape {}'.format(str(probs.shape)))
    if algorithm == 'threshold':
        n, t = params['n'], params['t']
        labels = [np.argmax(timechunk) for timechunk in probs]

        for i in range(len(probs)-n+1):
            # print(np.average(probs[i:i+n], axis=0)[0],
            #       np.average(probs[i:i+n], axis=0)[1])
            for c in range(probs.shape[-1]):
                avg = np.average(probs[i:i+n], axis=0)[c]
                if avg >= t:
                    # color.INFO('DEBUG',
                    #            'found threshold window of {} at [{}:{}] for class {}'.format(avg, i, i+n, c))
                    labels[i:i+n] = [c for _ in range(n)]

        return labels

    elif algorithm == 'hmm' or algorithm == 'viterbi':
        # define default emission probabilities
        default = {0: pmgt.DiscreteDistribution({'0' : 0.7, '1' : 0.3}),
                   1: pmgt.DiscreteDistribution({'0' : 0.2, '1' : 0.8})}

        states = []
        for c in [*range(probs.shape[-1])]:
            state = pmgt.State(params.get(c, default[c]), name=str(c))
            states += [state]

        model = pmgt.HiddenMarkovModel('laugh-decoder')
        model.add_states(states)

        if 'transitions' in params:
            model.add_transitions(params['transitions'])
        else:
            # start must always go to state 0
            model.add_transitions([model.start, states[0]],
                                  [states[0], model.end], [1., .1])
            model.add_transitions([states[0], states[0], states[1], states[1]],
                                  [states[0], states[1], states[0], states[1]],
                                  [.5, .4, .2, .8])
        model.bake()

        # if verbose:
        #     model.plot() # plotting is weird

        labels = [str(np.argmax(entry)) for entry in probs]
        labels = model.predict(sequence=labels, algorithm='viterbi')
        return labels[1:-1]

    else:
        raise NotImplementedError


def _detect_in_audio(wavdata, sr, model=None, precision=3,
                     algorithms=['threshold'], params=dict(n=5, t=.8),
                     verbose=True, norm=False, name='.noname'):
    '''
    This internal method is meant to tie together the two methods before it,
    `score_continuous_data`, and `decode_sequence`. The method takes in raw
    wave audio data, and does the heavylifting in detecting the sampling rate,
    assigning labels, and so on.
    ---
        wavdata:
        model: an instance of Keras BaseModel class that supports model.predict
               in order to assign multiclass/binary probabilities to data
        precision: the higher, the more precise (time-wise), and the slower it
                   takes to compute initially
        algorithms: a list of algorithms to use for scoring. results from using
                    all of the specified algorithms are returned
        params: dict; params specific to the algorithms requested. e.g., for
                threshold, n (window len) and t (threshold) may be supplied

        return: (decoded, preds), a tuple of dict with dict['algorithm'] storing
                the assigned labels according to 'algorithm', and preds storing
                the raw output from model.predict(), should the parent method
                need it for anything
    '''
    preds = score_continuous_data(wavdata=wavdata, sr=sr, model=model,
                                  precision=precision, norm=norm, name=name)

    decoded = defaultdict(list)
    for alg in algorithms:
        color.INFO('INFO', 'decoding labels with {}'.format(alg))
        try:
            decoded[alg] = decode_sequence(probs=preds, algorithm=alg,
                                           params=params, verbose=verbose)
        except NotImplementedError:
            color.INFO('FUTURE', 'WIP; {} not yet implemented'.format(alg))

    decoded['timestamp'] = [int(i*(.96e3/precision))
                            for i, _ in enumerate(preds)]

    return decoded, preds


def detect_in_episode(episode, model, precision=3, algorithms=['threshold'],
                      params=dict(n=5, t=.8), verbose=True, norm=False):
    '''
    wrapper for `_detect_in_audio` but tailored to episodes for this scheme of
    data
    ---
        episode: episode name in the standard naming scheme (friends-s%%-e%%)
        --
        for docstrings of the rest of the arguments, see `_detect_in_audio`
    '''
    sr, wavdata = wavfile.read('../wav/{}.wav'.format(episode))
    dec, preds = _detect_in_audio(wavdata=wavdata, sr=sr, model=model,
                                  precision=precision, algorithms=algorithms,
                                  params=params, verbose=verbose, norm=norm,
                                  name=episode)

    return dec, preds


def discrete_to_chunks(sequence=None):
    '''
    certain methods may give out discrete 0,1,... labels over time distributed
    data. this method will take such discrete labels and return a list of
    patches of each label in a dictionary with keys that are labels. labels
    must be hashable; in most cases labels would be integers denoting classes
    '''
    try:
        groups = [(k, len([*g])) for k, g in itertools.groupby(sequence)]
    except TypeError:
        color.ERR('ERR', 'please provide a valid discrete labels sequence')

    patches = defaultdict(list)
    offset = 0
    for k, wid in groups:
        patches[k] += [(offset, offset+wid)]
        offset += wid

    return patches
