#!/usr/bin/env python3
"""Simple Policy function module."""
import numpy as np


def policy(matrix, weight):
    """
    Compute the policy with a weight of a matrix.

    Args:
        matrix: numpy.ndarray representing the state
        weight: numpy.ndarray representing the weight

    Returns:
        numpy.ndarray of action probabilities (the policy)
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)
