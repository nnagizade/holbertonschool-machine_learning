#!/usr/bin/env python3
"""Creates, builds and trains a gensim word2vec model"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds and trains a gensim word2vec model

    Returns: the trained model
    """
    model = gensim.models.Word2Vec(vector_size=vector_size,
                                   min_count=min_count,
                                   window=window,
                                   negative=negative,
                                   sg=0 if cbow else 1,
                                   epochs=epochs,
                                   seed=seed,
                                   workers=workers,
                                   sorted_vocab=0)
    model.build_vocab(sentences)
    return model
