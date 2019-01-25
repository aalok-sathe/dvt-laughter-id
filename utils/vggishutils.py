#!/usr/bin/env python3

import sys
sys.path.append('../vggish/')

# local imports
import vggish_input
import vggish_slim
import vggish_postprocess
import color
import utils
# stdlib and package imports
import numpy as np
import tensorflow as tf


global sess
sess = None


def get_embed(input_wav, sess=sess):
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
    color.INFO('INFO', 'generating input example from wav')
    examples_batch = vggish_input.wavfile_to_examples(input_wav)

    # load models and postprocessor (a PCA model)
    color.INFO('INFO', 'loading vggish model checkpoint')
    pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    if sess == None:
        sess = tf.Session()
        tf.Graph().as_default()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
    else:
        color.INFO('INFO', 'attempting to reuse tensorflow session')

    color.INFO('INFO', 'generating features')
    features_tensor = sess.graph.get_tensor_by_name('vggish/input_features:0')
    color.INFO('INFO', 'generating embeddings')
    embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')

    # Compute embeddings:
    color.INFO('INFO', 'computing embeddings')
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    color.INFO('INFO', 'post-processing data')
    postprocessed_batch = pproc.postprocess(embedding_batch)

    # Print out dimensions:
    color.INFO('INFO', 'shape of input batches: ' + str(examples_batch.shape))
    color.INFO('INFO', 'shape of vggish output: ' + str(embedding_batch.shape))
    color.INFO('INFO', 'shape of postprocessed: '
                       + str(postprocessed_batch.shape))

    return postprocessed_batch, sess


# load the wave file
if __name__ == "__main__":
    input_wav = "/Users/taylor/local/dv/input/friends/friends-s02-e03.wav"
    get_embed(input_wav)
