#!/usr/bin/env python3
"""Calculates a GMM from a dataset using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d), dataset
        k: number of clusters

    Returns:
        pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,), cluster priors
        m: numpy.ndarray of shape (k, d), centroid means
        S: numpy.ndarray of shape (k, d, d), covariance matrices
        clss: numpy.ndarray of shape (n,), cluster indices for each
              data point
        bic: BIC value for the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
