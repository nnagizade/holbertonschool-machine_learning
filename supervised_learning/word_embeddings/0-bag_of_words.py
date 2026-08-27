#!/usr/bin/env python3
"""
Module for Bag of Words embedding matrix creation.
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    - sentences (list): List of sentences to analyze.
    - vocab (list, optional): List of vocabulary words to use.
      If None, all unique words across sentences are used.

    Returns:
    - embeddings (numpy.ndarray): Matrix of shape (s, f) containing word counts.
    - features (list): List of features used for embeddings.
    """
    cleaned_sentences = []
    all_words = set()

    for sentence in sentences:
        # Preprocess sentence: lowercase and strip possessive "'s"
        clean_sentence = sentence.lower().replace("'s", "")

        # Extract words by filtering out punctuation
        words = [
            word for word in ''.join(
                c if c.isalnum() else ' ' for c in clean_sentence
            ).split()
        ]

        cleaned_sentences.append(words)
        all_words.update(words)

    if vocab is None:
        features = sorted(list(all_words))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Map features to column indices for O(1) lookup
    feature_map = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feature_map:
                embeddings[i, feature_map[word]] += 1

    return embeddings, features
