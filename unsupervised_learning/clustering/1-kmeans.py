#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, max number of iterations

    Returns:
        C, clss or None, None on failure
        C: numpy.ndarray of shape (k, d), centroid means for each cluster
        clss: numpy.ndarray of shape (n,), index of cluster each point
              belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    if C is None:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)

    for i in range(iterations):
        C_prev = C.copy()

        # compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            points = X[clss == j]
            if points.shape[0] == 0:
                C[j] = np.random.uniform(low, high, size=(d,))
            else:
                C[j] = points.mean(axis=0)

        if np.array_equal(C, C_prev):
            distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
            clss = np.argmin(distances, axis=1)
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    return C, clss
