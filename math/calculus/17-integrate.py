#!/usr/bin/env python3
"""
Module to calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial
    Args:
        poly: list of coefficients representing a polynomial
        C: integer representing the integration constant
    Returns:
        New list of coefficients representing the integral
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    # The integral of an empty-logic list [0] is just [C]
    # But we start with C as the first element (x^0)
    integral = [C]

    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        
        # Power Rule: coefficient / (current_power + 1)
        value = poly[i] / (i + 1)
        
        # Requirement: Represent whole numbers as integers
        if value % 1 == 0:
            integral.append(int(value))
        else:
            integral.append(value)

    # Requirement: The returned list should be as small as possible
    # This removes trailing zeros that don't add value
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
