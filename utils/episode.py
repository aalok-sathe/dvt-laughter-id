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
import vggishutils
# stdlib and package imports
import yaml
from glob import glob
from collections import defaultdict
from scipy.io import wavfile


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
            return frame
        else:
            seconds = value / sr
            millisec = seconds * 1e3
            return millisec
    except (ValueError, TypeError) as e:
        color.ERR('ERR', 'please check your arguments')
        raise


def get_data(which_episodes=None, use_vggish=True, preserve_length=False):
    '''
    gets embeddings data for a list of episodes.
    as a list, expects basenames of the episodes without any attribute or
    filename extension.
    returns class data generated using patches loaded for the same episodes
    ---
        which_episodes: a list of basenames (e.g. "friends-s02-e03") of
                        episodes to process (required)
        use_vggish: whether to use vggish generating data (currently no
                    alternate method is implemented, but will be in the future)
        preserve_length: whether to return data as disjoint chunks of equal
                         length, or preserve length of annotated chunks and
                         return variable-length data (defaut: False)
    '''
    for ep in which_episodes:
        patches = load_annotations(ep)
        laughs = patches['laughter']
        nolaughs = [(s2-e1) for (s1, e1), (s2, e2) in zip(laughs, laughs[1:])
                    if s2-e1 > 1e3]
        rate, wavdata = wavfile.read('wav/{}.wav'.format(episode))

        X, Y = [], []

        # with open('episodes/{}-dvt.jsonl'.format(episode), 'r') as infile:
        #     fps = json.loads(infile.readline())['fps']
        lastend = 0

        for i, (start, end) in enumerate(laughs):
            if start == end: continue
            start_f, end_f = get_frame(start, fps), get_frame(end, fps)
            start_f, end_f = round(start_f*rate/fps), round(end_f*rate/fps)
            # print(start_f, end_f)
            try:
                this_X, utils.sess = get_embed(input_wav=wavdata[lastend:start_f], sr=rate, sess=sess)
                X.append(this_X)
                y.append(0)
            except tf.errors.InvalidArgumentError as e:
                print("encountered {}; resuming".format(e), file=sys.stderr)
                pass
            except ValueError as e:
                print("encountered {}; resuming".format(e), file=sys.stderr)
                pass
            except Exception as e:
                print("encountered {}; resuming".format(e), file=sys.stderr)
                pass
            try:
                this_X, sess = get_embed(input_wav=wavdata[start_f:end_f], sr=rate, sess=sess)
                X.append(this_X)
                y.append(1)
            except tf.errors.InvalidArgumentError as e:
                print("encountered {}; resuming".format(e), file=sys.stderr)
                pass
            except ValueError as e:
                print("encountered {}; resuming".format(e), file=sys.stderr)
                pass
            except Exception as e:
                print("encountered {}; resuming".format(e), file=sys.stderr)
                pass
            lastend = end_f
            print("processed {} of {}".format(i, len(patches)), end='\r')

    return X,y
