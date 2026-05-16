#!/usr/bin/env python3
"""
Module that defines the MultiNormal class with PDF calculation.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Class constructor for MultiNormal.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean: shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data: (data - mean)
        data_centered = data - self.mean

        # Calculate covariance matrix: shape (d, d)
        self.cov = (data_centered @ data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # 1. Calculate the determinant and inverse of the covariance matrix
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # 2. Calculate the normalization constant (denominator)
        # (2 * pi)^(d/2) * sqrt(det)
        denom = np.sqrt(((2 * np.pi) ** d) * det)

        # 3. Calculate the exponent part: -0.5 * (x - mu).T @ inv @ (x - mu)
        x_m = x - self.mean
        exponent = -0.5 * (x_m.T @ inv @ x_m)

        # 4. Final PDF value
        pdf_val = (1.0 / denom) * np.exp(exponent)

        # Returns as a float value from the 1x1 result matrix
        return pdf_val[0][0]
