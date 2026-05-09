#!/usr/bin/env python3
"""
Contains the Normal class with PDF method
"""


class Normal:
    """
    Class that represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            sum_diff_sq = sum([(x - self.mean) ** 2 for x in data])
            self.stddev = float((sum_diff_sq / len(data)) ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        Args:
            x: the x-value
        Returns:
            The PDF value for x
        """
        e = 2.7182818285
        pi = 3.1415926536
        mean = self.mean
        stddev = self.stddev

        # Breakdown of the Normal PDF formula
        exponent = -0.5 * ((x - mean) / stddev) ** 2
        coefficient = 1 / (stddev * ((2 * pi) ** 0.5))

        pdf_val = coefficient * (e ** exponent)

        return pdf_val
