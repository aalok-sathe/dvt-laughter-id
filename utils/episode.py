#! /bin/env/ python3
'''
this file is intended to house util functions specific to handling an
individual episode.
for an episode 'episode_i', we make these assumptions:
    - ../video/episode_i.mp4
    - ../video/episode_i-dvt.jsonl
    - ../wav/episode_i.wav
'''

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


def get_data(which_episodes=None, use_vggish=True, preserve_length=False):
    '''
    gets embeddings data for a list of episodes.
    as a list, expects basenames of the episodes without any attribute or
    filename extension.
    returns class data generated using patches loaded for the same episodes.
    '''
    for ep in which_episodes:
        patches = load_annotations(ep)
        laughs = patches['laughter']
        rate, wavdata = wavfile.read('wav/{}.wav'.format(episode))

        X, Y = [], []
    with open('episodes/{}-dvt.jsonl'.format(episode), 'r') as infile:
        fps = json.loads(infile.readline())['fps']
    lastend = 0
    for i, (start, end, attrib) in enumerate(patches):
        if start == end: continue
        global sess
        start_f, end_f = get_frame(start, fps), get_frame(end, fps)
        start_f, end_f = round(start_f*rate/fps), round(end_f*rate/fps)
        # print(start_f, end_f)
        try:
            this_X, sess = get_embed(input_wav=wavdata[lastend:start_f], sr=rate, sess=sess)
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
