#!/usr/bin/env python3
"""
Module to shuffle two matrices in the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
            m: number of data points
            nx: number of features in X
        Y: numpy.ndarray of shape (m, ny) to shuffle
            m: same number of data points as in X
            ny: number of features in Y

    Returns:
        The shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
