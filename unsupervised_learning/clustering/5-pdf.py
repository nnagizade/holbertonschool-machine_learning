#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian
    distribution.

    Args:
        X: numpy.ndarray of shape (n, d), data points whose PDF
           should be evaluated
        m: numpy.ndarray of shape (d,), mean of the distribution
        S: numpy.ndarray of shape (d, d), covariance of the
           distribution

    Returns:
        P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values
        for each data point, with a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    X_m = X - m

    exponent = -0.5 * np.sum(X_m @ inv * X_m, axis=1)

    denom = np.sqrt(((2 * np.pi) ** d) * det)

    P = np.exp(exponent) / denom

    P = np.maximum(P, 1e-300)

    return P
