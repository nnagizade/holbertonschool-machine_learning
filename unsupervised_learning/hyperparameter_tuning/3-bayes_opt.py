#!/usr/bin/env python3
"""Initializes Bayesian optimization on a noiseless 1D Gaussian process"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian
    process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor.

        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1), inputs already
                    sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1), outputs of the
                    black-box function for each input in X_init
            bounds: tuple of (min, max), bounds of the space in
                    which to look for the optimal point
            ac_samples: number of samples that should be analyzed
                        during acquisition
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                     black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool, determines whether optimization should
                      be performed for minimization (True) or
                      maximization (False)

        Sets the public instance attributes f, gp, X_s, xsi, minimize
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_b, max_b = bounds
        self.X_s = np.linspace(min_b, max_b, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
