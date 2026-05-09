#!/usr/bin/env python3
"""
Contains the Exponential class which represents an exponential distribution
"""


class Exponential:
    """
    Class that represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Class constructor
        Args:
            data: a list of the data to be used to estimate the distribution
            lambtha: the expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Lambtha for exponential is 1 / mean(data)
            # mean = sum(data) / len(data)
            # lambtha = 1 / mean -> len(data) / sum(data)
            self.lambtha = float(len(data) / sum(data))
