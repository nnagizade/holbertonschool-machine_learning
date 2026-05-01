#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    Args:
        matrix: list of lists whose inverse should be calculated
    Returns:
        The inverse of matrix, or None if matrix is singular
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

    # Import determinant from task 0
    det_func = __import__('0-determinant').determinant
    det = det_func(matrix)

    # If determinant is 0, the matrix is singular
    if det == 0:
        return None

    # Special case for 1x1 matrix: inverse is 1/element
    if n == 1:
        return [[1 / matrix[0][0]]]

    # Import adjugate from task 3
    adj_func = __import__('3-adjugate').adjugate
    adj_matrix = adj_func(matrix)

    # Multiply adjugate by 1/det
    inv_matrix = []
    for i in range(n):
        inv_row = []
        for j in range(n):
            inv_row.append(adj_matrix[i][j] / det)
        inv_matrix.append(inv_row)

    return inv_matrix
