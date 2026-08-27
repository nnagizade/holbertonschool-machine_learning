#!/usr/bin/env python3
"""Initializes a noiseless 1D Gaussian process and predicts mean/std"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor.

        Args:
            X_init: numpy.ndarray of shape (t, 1), inputs already
                    sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1), outputs of the
                    black-box function for each input in X_init
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                     black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two
        matrices using the Radial Basis Function (RBF).

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            the covariance kernel matrix as a numpy.ndarray of shape
            (m, n)
        """
        a = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        b = np.sum(X2 ** 2, axis=1)
        c = 2 * np.dot(X1, X2.T)
        sqdist = a + b - c

        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a
        Gaussian process.

        Args:
            X_s: numpy.ndarray of shape (s, 1), points whose mean
                 and standard deviation should be calculated

        Returns:
            mu, sigma
            mu: numpy.ndarray of shape (s,), mean for each point
                in X_s
            sigma: numpy.ndarray of shape (s,), variance for each
                   point in X_s
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu = K_s.T.dot(K_inv).dot(self.Y)
        mu = mu.reshape(-1)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)

        return mu, sigma
