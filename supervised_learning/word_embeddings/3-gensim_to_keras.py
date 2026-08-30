#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer

    model is a trained gensim word2vec model
    Returns: the trainable keras Embedding
    """
    vectors = model.wv.vectors
    return tf.keras.layers.Embedding(input_dim=vectors.shape[0],
                                     output_dim=vectors.shape[1],
                                     weights=[vectors],
                                     trainable=True)
