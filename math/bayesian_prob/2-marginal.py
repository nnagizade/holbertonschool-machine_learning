#!/usr/bin/env python3
"""
Module to calculate the marginal probability of obtaining data.
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining data.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities
        Pr: 1D numpy.ndarray containing the prior beliefs of P

    Returns:
        The marginal probability of obtaining x and n
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate Binomial Coefficient
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    coeff = fact_n / (fact_x * fact_nx)

    # Calculate Likelihood: P(x | P)
    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))

    # Calculate Intersection: P(x | P) * P(P)
    intersection = likelihood * Pr

    # Marginal Probability is the sum of all intersections
    marginal_prob = np.sum(intersection)

    return marginal_prob
