#!/usr/bin/env python3
"""
Module to normalize (standardize) a matrix
"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
            d: number of data points
            nx: number of features
        m: numpy.ndarray of shape (nx,) containing the mean of features
        s: numpy.ndarray of shape (nx,) containing standard deviation

    Returns:
        The normalized X matrix
    """
    return (X - m) / s
