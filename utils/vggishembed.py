#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import sys

import vggish_input
import vggish_slim
import vggish_postprocess

def get_embed(input_wav):
    print("generating examples_batch", file=sys.stderr)
    examples_batch = vggish_input.wavfile_to_examples(input_wav)

    # load models and postprocessor (a PCA model)
    print("loading vggish model checkpoint", file=sys.stderr)
    pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    sess = tf.Session()
    tf.Graph().as_default()
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
    print("generating features", file=sys.stderr)
    features_tensor = sess.graph.get_tensor_by_name('vggish/input_features:0')
    print("identifying embeddings", file=sys.stderr)
    embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')

    # Compute embeddings:
    print("generating embeddings", file=sys.stderr)
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    print("post-processing", file=sys.stderr)
    postprocessed_batch = pproc.postprocess(embedding_batch)

    # Print out dimensions:
    print("Shape of input batches: " + str(examples_batch.shape), file=sys.stderr)
    print("Shape of output from vggish: " + str(embedding_batch.shape), file=sys.stderr)
    print("Shape of output from postprocess: " + str(postprocessed_batch.shape), file=sys.stderr)

    return postprocessed_batch

# load the wave file
if __name__ == "__main__":
    input_wav = "/Users/taylor/local/dv/input/friends/friends-s02-e03.wav"
    get_embed(input_wav)
