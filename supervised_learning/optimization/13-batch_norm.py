#!/usr/bin/env python3
"""
Module to normalize an unactivated output of a neural network
using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n) that should be normalized
           m is the number of data points
           n is the number of features in Z
        gamma: numpy.ndarray of shape (1, n) containing scales
        beta: numpy.ndarray of shape (1, n) containing offsets
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    # Calculate mean along columns (features)
    mean = np.mean(Z, axis=0, keepdims=True)

    # Calculate variance along columns
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift using gamma and beta
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
