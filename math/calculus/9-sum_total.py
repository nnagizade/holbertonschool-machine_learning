#!/usr/bin/env python3
"""
Module to calculate the sum of i squared from 1 to n
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares up to n without loops
    Args:
        n: the stopping condition
    Returns:
        The integer value of the sum, or None if n is invalid
    """
    if not isinstance(n, int) or n < 1:
        return None
    # Using the formula: n(n + 1)(2n + 1) / 6
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
