#!/usr/bin/env python3
"""Module to train Word2Vec model using gensim."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim word2vec model."""
    sg = 0 if cbow else 1

    if gensim.__version__[0] < '4':
        return gensim.models.Word2Vec(
            sentences=sentences,
            size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg,
            iter=epochs,
            seed=seed,
            workers=workers
        )

    return gensim.models.Word2Vec(
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
