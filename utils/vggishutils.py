#!/usr/bin/env python3

import sys
sys.path.append('../vggish/')

# local imports
import utils
import vggish_input
import vggish_slim
import vggish_postprocess
import color
# stdlib and package imports
import numpy as np
import tensorflow as tf


def get_embed(input_wav, sr=None, sess=None):
    '''
    accepts an input of raw wav data and produces an embedding for it treating
    the entire wav data as one sequence of audio
    ---
        input_wav: raw wav data as an numpy ndarray
        sess: existing tensorflow if already active (required) or None if not

        return: postprocessed_batch (the embeddings), and sess, the tf session
                used so that it can be reused. note that returned sess must be
                handled appropriately by the user
    '''
    # color.INFO('INFO', 'generating input example from wav\r')
    examples_batch = vggish_input.waveform_to_examples(input_wav, sr)

    # load models and postprocessor (a PCA model)
    # color.INFO('INFO', 'loading vggish model checkpoint\r')
    pproc = vggish_postprocess.Postprocessor('../vggish/vggish_pca_params.npz')
    if sess == None:
        sess = tf.Session()
        tf.Graph().as_default()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, '../vggish/vggish_model.ckpt')
    else:
        # color.INFO('INFO', 'attempting to reuse tensorflow session\r')
        pass

    # color.INFO('INFO', 'generating features\r')
    features_tensor = sess.graph.get_tensor_by_name('vggish/input_features:0')
    # color.INFO('INFO', 'generating embeddings\r')
    embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')

    # Compute embeddings:
    # color.INFO('INFO', 'computing embeddings\r')
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    # color.INFO('INFO', 'post-processing data\r')
    postprocessed_batch = pproc.postprocess(embedding_batch)

    # Print out dimensions: # TODO: make str formatting error go away
    # color.INFO('INFO', 'shape of input batches: %s\r' % examples_batch.shape)
    # color.INFO('INFO', 'shape of vggish output: %s\r' % embedding_batch.shape)
    # color.INFO('INFO', 'shape postprocessed: %s\r' % postprocessed_batch.shape)

    return postprocessed_batch, sess


# load the wave file
# if __name__ == "__main__":
#     input_wav = "/Users/taylor/local/dv/input/friends/friends-s02-e03.wav"
#     get_embed(input_wav)
