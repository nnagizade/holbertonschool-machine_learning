#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    Args:
        matrix: list of lists whose cofactor matrix should be calculated
    Returns:
        The cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Reuse the minor logic from task 1
    minor_func = __import__('1-minor').minor
    minor_matrix = minor_func(matrix)

    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Apply checkerboard sign: (-1)^(i + j)
            # If i+j is even, sign is positive (+1)
            # If i+j is odd, sign is negative (-1)
            sign = (-1) ** (i + j)
            cofactor_row.append(minor_matrix[i][j] * sign)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
