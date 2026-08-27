#!/usr/bin/env python3
"""Performs K-means on a dataset using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d), dataset
        k: number of clusters

    Returns:
        C, clss
        C: numpy.ndarray of shape (k, d), centroid means for each
           cluster
        clss: numpy.ndarray of shape (n,), index of the cluster in C
              that each data point belongs to
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
