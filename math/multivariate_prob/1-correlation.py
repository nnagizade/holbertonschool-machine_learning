#!/usr/bin/env python3
"""
Module to calculate a correlation matrix from a covariance matrix.
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Extract variances from the diagonal
    diag = np.diag(C)

    # Standard deviation is the square root of the variance
    std_dev = np.sqrt(diag)

    # To calculate correlation matrix:
    # corr = covariance / (std_dev_row * std_dev_col)
    # Using outer product to create the matrix of (std_dev_i * std_dev_j)
    outer_std = np.outer(std_dev, std_dev)

    correlation_matrix = C / outer_std

    return correlation_matrix
