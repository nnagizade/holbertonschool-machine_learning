#!/usr/bin/env python3
"""Initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters

    Returns:
        numpy.ndarray of shape (k, d) with initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    low = X.min(axis=0)
    high = X.max(axis=0)

    centroids = np.random.uniform(low, high, size=(k, d))

    return centroids
