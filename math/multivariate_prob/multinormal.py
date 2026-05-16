#!/usr/bin/env python3
"""
Module that defines the MultiNormal class.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Class constructor for MultiNormal.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                d: number of dimensions
                n: number of data points
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean: shape (d, 1)
        # axis=1 calculates mean across the data points for each dimension
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data: (data - mean)
        data_centered = data - self.mean

        # Calculate covariance matrix: shape (d, d)
        # Since data is (d, n), we do (data_centered @ data_centered.T)
        self.cov = (data_centered @ data_centered.T) / (n - 1)
