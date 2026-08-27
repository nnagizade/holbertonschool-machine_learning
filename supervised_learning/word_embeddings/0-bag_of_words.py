#!/usr/bin/env python3
"""
Module to create a Bag of Words embedding matrix.
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    - sentences (list): List of sentences to analyze.
    - vocab (list, optional): List of vocabulary words to use.

    Returns:
    - embeddings (numpy.ndarray): Matrix of shape (s, f).
    - features (numpy.ndarray): Array of features used.
    """
    cleaned_sentences = []
    all_words = set()

    for sentence in sentences:
        clean_s = sentence.lower().replace("'s", "")
        clean_s = "".join(c if c.isalnum() else " " for c in clean_s)
        words = clean_s.split()

        cleaned_sentences.append(words)
        all_words.update(words)

    if vocab is None:
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    features = np.array(features)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    feature_map = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feature_map:
                embeddings[i, feature_map[word]] += 1

    return embeddings, features
