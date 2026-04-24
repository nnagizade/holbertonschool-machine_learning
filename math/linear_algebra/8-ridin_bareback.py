#!/usr/bin/env python3
"""Defines a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication and returns a new matrix"""
    # Check if matrices can be multiplied (m x n * n x p)
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    # Shape of result will be len(mat1) x len(mat2[0])
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
