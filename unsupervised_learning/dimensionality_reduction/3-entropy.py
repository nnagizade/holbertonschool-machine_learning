#!/usr/bin/env python3
"""Defines a function that calculates Shannon entropy and P affinities."""
import numpy as np


def HP(Di, beta):
    """
    Calculate the Shannon entropy and P affinities relative to a data
    point.

    Di: a numpy.ndarray of shape (n - 1,) containing the pariwise
        distances between a data point and all other points except
        itself
        n is the number of data points
    beta: a numpy.ndarray of shape (1,) containing the beta value for
        the Gaussian distribution

    Returns: (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,) containing the P
            affinities of the points
    """
    Pi = np.exp(-Di * beta)
    Pi = Pi / np.sum(Pi)

    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
