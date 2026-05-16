#!/usr/bin/env python3
"""
Module to calculate the likelihood of obtaining data
following a binomial distribution.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various
    hypothetical probabilities.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities

    Returns:
        1D numpy.ndarray containing the likelihood for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate binomial coefficient: n! / (x! * (n - x)!)
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    coefficient = fact_n / (fact_x * fact_nx)

    # Calculate likelihood for each p in P
    # Likelihood = (n choose x) * (p^x) * ((1-p)^(n-x))
    like = coefficient * (P ** x) * ((1 - P) ** (n - x))

    return like
