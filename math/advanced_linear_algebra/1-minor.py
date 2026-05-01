#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    Args:
        matrix: list of lists whose minor matrix should be calculated
    Returns:
        The minor matrix of matrix
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

    # Special case for 1x1 matrix: the minor is defined as 1
    if n == 1:
        return [[1]]

    # Import determinant from task 0
    det_func = __import__('0-determinant').determinant

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create sub-matrix by removing row i and column j
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] +
                                                          matrix[i+1:])]
            minor_row.append(det_func(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
