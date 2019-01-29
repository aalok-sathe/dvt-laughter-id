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
# stdlib and package imports
import yaml
from glob import glob
from collections import defaultdict
from scipy.io import wavfile
import numpy as np
from progressbar import progressbar
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


def convertref(value=None, sr=None, to='audio'):
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


def get_data(which_episodes=None, use_vggish=True, preserve_length=False,
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

        patches = load_annotations(ep)
        laughs = patches[task]
        nolaughs = [(e1, s2) for (s1, e1), (s2, e2) in zip(laughs, laughs[1:])
                    if s2-e1 > 1e3]

        existsflag = False
        archivepath = Path(archive)
        archivepath = archivepath.joinpath(ep + '_%s_datachunks.npz' % task)

        if archive:
            # check if archives already exist
            if archivepath.exists():
                color.INFO('INFO', 'loading from {}'.format(archivepath))
                existsflag = True
                arrays = np.load(archivepath)
                this_X = arrays['X'].to_list()
                this_Y = arrays['Y'].to_list()
                this_refs = arrays['refs'].to_list()
            else:
                this_X, this_Y, this_refs = [], [], []
                sr, wavdata = wavfile.read('../wav/{}.wav'.format(ep))
        else:
            sr, wavdata = wavfile.read('../wav/{}.wav'.format(ep))

        color.INFO('INFO', 'processing %s data in %s' % (task, ep))
        for start, end in progressbar(laughs, redirect_stdout=1):
            if existsflag: break
            if start == end: continue
            start_f, end_f = convertref(start, sr), convertref(end, sr)
            # print(start_f, end_f)
            try:
                this_x, utils.sess = get_embed(input_wav=wavdata[start_f:end_f],
                                     sr=sr, sess=utils.sess)
                if preserve_length:
                    this_X += [this_x]
                    this_Y += [1]
                    this_refs += '{} {} {}'.format(ep, start, end)
                else:
                    this_X += [x.reshape(1, -1) for x in this_x]
                    this_Y += [1 for _ in this_x]
                    this_refs += ['{} {} {}'.format(ep, start, end)
                                  for _ in this_x]
            # except (tf.errors.InvalidArgumentError, Exception) as e:
            except Exception as e:
                color.ERR('INFO', 'encountered {}; resuming...\r'.format(e))
                pass

        color.INFO('INFO', 'processing no-%s data in %s' % (task, ep))
        for start, end in progressbar(nolaughs, redirect_stdout=1):
            if existsflag: break
            if start == end: continue
            start_f, end_f = convertref(start, sr), convertref(end, sr)
            # print(start_f, end_f)
            try:
                this_x, utils.sess = get_embed(input_wav=wavdata[start_f:end_f],
                                     sr=sr, sess=utils.sess)
                if preserve_length:
                    this_X += [this_x]
                    this_Y += [0]
                    this_refs += '{} {} {}'.format(ep, start, end)
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
        this_refs = np.array(this_refs, dtype=str)

        if archive and not existsflag:
            np.savez_compressed(archivepath, X=this_X, Y=this_Y, refs=this_refs)

        del this_X; del this_Y; del this_refs

    if preserve_length:
        return X, Y, refs
    else:
        return np.vstack(X), np.array(Y, dtype=int), np.array(refs, dtype=str)
