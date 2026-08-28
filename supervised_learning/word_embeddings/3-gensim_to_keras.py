#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: a trained gensim word2vec model

    Returns:
        the trainable keras Embedding layer
    """
    vectors = model.wv.vectors
    vocab_size, embedding_dim = vectors.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[vectors],
        trainable=True
    )

    return embedding_layer
