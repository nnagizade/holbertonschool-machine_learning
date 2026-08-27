#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the Bayesian
Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d), data set
        kmin: positive integer, minimum number of clusters to check
              for (inclusive)
        kmax: positive integer, maximum number of clusters to check
              for (inclusive)
        iterations: positive integer, max number of iterations for
                    the EM algorithm
        tol: non-negative float, tolerance for the EM algorithm
        verbose: boolean, determines if the EM algorithm should
                 print information to standard output

    Returns:
        best_k, best_result, l, b, or None, None, None, None on
        failure
        best_k: best value for k based on its BIC
        best_result: tuple containing pi, m, S
        l: numpy.ndarray of shape (kmax - kmin + 1), log likelihood
           for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1), BIC value for
           each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = range(kmin, kmax + 1)
    l_list = []
    b_list = []
    results = []

    for k in ks:
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None:
            return None, None, None, None

        results.append((pi, m, S))
        l_list.append(log_l)

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        bic = p * np.log(n) - 2 * log_l
        b_list.append(bic)

    l_arr = np.array(l_list)
    b_arr = np.array(b_list)

    best_i = np.argmin(b_arr)
    best_k = kmin + best_i
    best_result = results[best_i]

    return best_k, best_result, l_arr, b_arr
