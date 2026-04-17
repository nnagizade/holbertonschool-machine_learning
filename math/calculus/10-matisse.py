#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    Args:
        poly: list of coefficients representing a polynomial
    Returns:
        New list of coefficients representing the derivative
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # If the polynomial is just a constant (e.g., [5]), derivative is [0]
    if len(poly) == 1:
        return [0]

    derivative = []
    # We skip index 0 because the derivative of a constant is always 0
    for i in range(1, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        # Power Rule: coefficient * power (which is the index i)
        derivative.append(poly[i] * i)

    return derivative
