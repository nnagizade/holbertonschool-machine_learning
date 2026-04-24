#!/usr/bin/env python3
"""Defines a function that performs element-wise operations on numpy.ndarrays"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and division"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
