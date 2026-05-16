#!/usr/bin/env python3
"""
Module to calculate the posterior probability for various
hypothetical probabilities.
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for various hypothetical
    probabilities of developing severe side effects.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities
        Pr: 1D numpy.ndarray containing the prior beliefs of P

    Returns:
        1D numpy.ndarray containing the posterior probability
        for each probability in P
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

    # 1. Calculate Binomial Coefficient
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    coeff = fact_n / (fact_x * fact_nx)

    # 2. Calculate Likelihood: P(data | p)
    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))

    # 3. Calculate Intersection (Numerator): Likelihood * Prior
    intersection = likelihood * Pr

    # 4. Calculate Marginal (Denominator): Sum of all intersections
    marginal = np.sum(intersection)

    # 5. Calculate Posterior: Intersection / Marginal
    post = intersection / marginal

    return post
