#!/usr/bin/env python3
"""Module for training Word2Vec models using gensim."""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim word2vec model.

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum occurrences of a word for use in training
        window: max distance between current and predicted word
        negative: size of negative sampling
        cbow: boolean for training type (True for CBOW, False for Skip-gram)
        epochs: number of iterations to train over
        seed: seed for random number generator
        workers: number of worker threads

    Returns:
        The trained Word2Vec model
    """
    sg = 0 if cbow else 1
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
