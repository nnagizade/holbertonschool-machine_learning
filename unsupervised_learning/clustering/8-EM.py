#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                               verbose=False):
    """Performs the expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d), data set
        k: positive integer, number of clusters
        iterations: positive integer, max number of iterations
        tol: non-negative float, tolerance of log likelihood for
             early stopping
        verbose: boolean, whether to print information about the
                 algorithm

    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,), priors for each cluster
        m: numpy.ndarray of shape (k, d), centroid means for each
           cluster
        S: numpy.ndarray of shape (k, d, d), covariance matrices for
           each cluster
        g: numpy.ndarray of shape (k, n), probabilities for each
           data point in each cluster
        l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, prev_l = expectation(X, pi, m, S)
    if g is None or prev_l is None:
        return None, None, None, None, None

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            0, prev_l.round(5)))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or abs(l - prev_l) <= tol):
            print('Log Likelihood after {} iterations: {}'.format(
                i, l.round(5)))

        if abs(l - prev_l) <= tol:
            break

        prev_l = l

    return pi, m, S, g, l
