#!/usr/bin/env python3
"""Defines a function that calculates the cost of the t-SNE transformation."""
import numpy as np


def cost(P, Q):
    """
    Calculate the cost of the t-SNE transformation.

    P: a numpy.ndarray of shape (n, n) containing the P affinities
    Q: a numpy.ndarray of shape (n, n) containing the Q affinities

    Returns: C, the cost of the transformation
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    C = np.sum(P * np.log(P / Q))

    return C
