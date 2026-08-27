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
    - vocab (list, optional): List of vocabulary words to use for analysis.
      If None, all words within sentences are used.

    Returns:
    - embeddings (numpy.ndarray): Matrix of shape (s, f) with word counts.
    - features (numpy.ndarray): Array of features used for embeddings.
    """
    cleaned_sentences = []
    all_words = set()

    for sentence in sentences:
        # Lowercase and remove possessive 's
        clean_s = sentence.lower().replace("'s", "")
        # Replace non-alphanumeric characters with spaces
        clean_s = "".join(c if c.isalnum() else " " for c in clean_s)
        words = clean_s.split()
        
        cleaned_sentences.append(words)
        all_words.update(words)

    if vocab is None:
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    # Convert features to a NumPy array for correct stdout formatting
    features = np.array(features)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Hash map for O(1) feature index lookups
    feature_map = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feature_map:
                embeddings[i, feature_map[word]] += 1

    return embeddings, features
