#!/usr/bin/env python3
"""
Contains the Poisson class which represents a Poisson distribution
"""


class Poisson:
    """
    Class that represents a poisson distribution
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
            
            # Lambtha of a Poisson distribution is the mean of the data
            self.lambtha = float(sum(data) / len(data))
