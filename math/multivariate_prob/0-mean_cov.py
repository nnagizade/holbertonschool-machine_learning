#!/usr/bin/env python3
"""
Module to calculate the mean and covariance of a data set.
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
            n: number of data points
            d: number of dimensions

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate mean along axis 0 (rows), keep dims for shape (1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data: X - mean
    X_centered = X - mean

    # Calculate Covariance: (1 / (n - 1)) * (X_centered.T @ X_centered)
    # Using dot product @ is the same as (X - mu).T * (X - mu)
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
