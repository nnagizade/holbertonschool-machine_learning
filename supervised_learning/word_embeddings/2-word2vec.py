#!/usr/bin/env python3
"""Creates, builds, and trains a gensim word2vec model"""
os = __import__('os')
sys = __import__('sys')

if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                    negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim word2vec model

    Args:
        sentences: a list of sentences to be trained on
        vector_size: the dimensionality of the embedding layer
        min_count: the minimum number of occurrences of a word for use
            in training
        window: the maximum distance between the current and predicted
            word within a sentence
        negative: the size of negative sampling
        cbow: a boolean to determine the training type; True is for
            CBOW; False is for Skip-gram
        epochs: the number of iterations to train over
        seed: the seed for the random number generator
        workers: the number of worker threads to train the model

    Returns:
        the trained model
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers)

    return model
