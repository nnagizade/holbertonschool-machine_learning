#!/usr/bin/env python3
"""
Module to compute TF-IDF embeddings using TfidfVectorizer.
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Parameters:
    - sentences (list): List of sentences to analyze.
    - vocab (list, optional): List of vocabulary words to use.

    Returns:
    - embeddings (numpy.ndarray): Matrix of shape (s, f).
    - features (numpy.ndarray): Array of features used.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
