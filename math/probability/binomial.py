#!/usr/bin/env python3
"""
Contains the Binomial class with CDF method
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial distribution"""
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
            
            mean = sum(data) / len(data)
            sum_diff_sq = sum([(x - mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)
            p_est = 1 - (variance / mean)
            self.n = int(round(mean / p_est))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(num):
            res = 1
            for i in range(1, num + 1):
                res *= i
            return res

        n = self.n
        p = self.p
        comb = factorial(n) / (factorial(k) * factorial(n - k))
        return comb * (p ** k) * ((1 - p) ** (n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of 'successes'
        Args:
            k: the number of successes
        Returns:
            The CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1.0

        # Sum of PMF values from 0 to k
        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)

        return cdf_val
