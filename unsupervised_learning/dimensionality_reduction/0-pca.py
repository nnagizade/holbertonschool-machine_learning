#!/usr/bin/env python3
"""Defines a function that performs PCA on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on a dataset.

    X: a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    var: the fraction of the variance that the PCA transformation
        should maintain

    Returns: the weights matrix, W, that maintains var fraction of
        X's original variance
        W is a numpy.ndarray of shape (d, nd) where nd is the new
        dimensionality of the transformed X
    """
    u, s, vh = np.linalg.svd(X)

    cumsum = np.cumsum(s)
    total = cumsum[-1]

    threshold = cumsum / total

    nd = np.argwhere(threshold >= var)[0, 0] + 1

    W = vh[:nd].T

    return W 
