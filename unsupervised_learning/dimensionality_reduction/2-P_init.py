#!/usr/bin/env python3
"""Defines a function that initializes variables required for t-SNE."""
import numpy as np


def P_init(X, perplexity):
    """
    Initialize all variables required to calculate the P affinities in
    t-SNE.

    X: a numpy.ndarray of shape (n, d) containing the dataset to be
        transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    perplexity: the perplexity that all Gaussian distributions should
        have

    Returns: (D, P, betas, H)
        D: a numpy.ndarray of shape (n, n) that calculates the squared
            pairwise distance between two data points, with the
            diagonal set to 0
        P: a numpy.ndarray of shape (n, n) initialized to all 0's that
            will contain the P affinities
        betas: a numpy.ndarray of shape (n, 1) initialized to all 1's
            that will contain all of the beta values
        H is the Shannon entropy for perplexity with a base of 2
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)

    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
