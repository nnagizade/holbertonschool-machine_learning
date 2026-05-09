#!/usr/bin/env python3
"""
Contains the Poisson class with PMF method
"""


class Poisson:
    """
    Class that represents a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initialize Poisson distribution
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'
        Args:
            k: the number of successes
        Returns:
            The PMF value for k
        """
        # Convert k to integer as requested
        k = int(k)

        # Poisson distribution is only defined for k >= 0
        if k < 0:
            return 0

        # Mathematical constants
        e = 2.7182818285
        lambtha = self.lambtha

        # Calculate factorial of k
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # PMF Formula: (e^-lambda * lambda^k) / k!
        pmf_val = (e ** (-lambtha) * (lambtha ** k)) / factorial

        return pmf_val
