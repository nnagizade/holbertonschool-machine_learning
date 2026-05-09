#!/usr/bin/env python3
"""
Contains the Binomial class which represents a binomial distribution
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor
        Args:
            data: a list of the data to be used to estimate the distribution
            n: the number of Bernoulli trials
            p: the probability of a 'success'
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate Mean and Variance
            mean = sum(data) / len(data)
            sum_diff_sq = sum([(x - mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)

            # Estimate p and n
            # Variance = n*p*(1-p), Mean = n*p
            # Variance / Mean = 1 - p  => p = 1 - (Var / Mean)
            estimated_p = 1 - (variance / mean)
            
            # Calculate n and round to nearest integer
            estimated_n = round(mean / estimated_p)
            
            # Recalculate p based on the rounded n
            self.n = int(estimated_n)
            self.p = float(mean / self.n)
