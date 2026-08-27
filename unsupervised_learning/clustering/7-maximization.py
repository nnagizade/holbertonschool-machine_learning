#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d), data set
        g: numpy.ndarray of shape (k, n), posterior probabilities for
           each data point in each cluster

    Returns:
        pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,), updated priors for each
            cluster
        m: numpy.ndarray of shape (k, d), updated centroid means for
           each cluster
        S: numpy.ndarray of shape (k, d, d), updated covariance
           matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    pi = np.sum(g, axis=1) / n

    m = np.matmul(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        X_m = X - m[i]
        S[i] = np.matmul(g[i] * X_m.T, X_m) / np.sum(g[i])

    return pi, m, S
