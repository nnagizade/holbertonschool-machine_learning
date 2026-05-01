#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a matrix
"""


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    Args:
        matrix: list of lists whose adjugate matrix should be calculated
    Returns:
        The adjugate matrix of matrix
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

    # Reuse the cofactor logic from task 2
    cofactor_func = __import__('2-cofactor').cofactor
    cofact_mat = cofactor_func(matrix)

    # Transpose the cofactor matrix to get the adjugate
    adj_matrix = []
    for j in range(n):
        new_row = []
        for i in range(n):
            new_row.append(cofact_mat[i][j])
        adj_matrix.append(new_row)

    return adj_matrix
