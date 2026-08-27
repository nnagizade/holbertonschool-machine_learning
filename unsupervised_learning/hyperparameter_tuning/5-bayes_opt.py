#!/usr/bin/env python3
"""Initializes and performs Bayesian optimization on a noiseless 1D
Gaussian process"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """Calculates the next best sample location using the
        Expected Improvement acquisition function.
        Returns:
            X_next, EI
            X_next: numpy.ndarray of shape (1,), next best sample
                    point
            EI: numpy.ndarray of shape (ac_samples,), expected
                improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = mu - f_best - self.xsi
        with np.errstate(divide='warn'):
            Z = np.zeros_like(imp)
            mask = sigma != 0
            Z[mask] = imp[mask] / sigma[mask]
            EI = np.zeros_like(imp)
            EI[mask] = (imp[mask] * norm.cdf(Z[mask])
                        + sigma[mask] * norm.pdf(Z[mask]))
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function.
        Args:
            iterations: maximum number of iterations to perform
        Returns:
            X_opt, Y_opt
            X_opt: numpy.ndarray of shape (1,), optimal point
            Y_opt: numpy.ndarray of shape (1,), optimal function
                   value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        self.gp.X = self.gp.X[:-1]
        self.gp.Y = self.gp.Y[:-1]

        return X_opt, Y_opt
