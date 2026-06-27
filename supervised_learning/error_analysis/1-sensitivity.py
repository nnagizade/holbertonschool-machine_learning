#!/usr/bin/env python3
"""
Module to calculate the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): a confusion matrix of shape (classes, classes)
                                   where row indices represent correct labels
                                   and column indices represent predicted labels

    Returns:
        numpy.ndarray: a numpy.ndarray of shape (classes,) containing the
                       sensitivity of each class
    """
    # True Positives are the diagonal elements
    true_positives = np.diagonal(confusion)

    # Actual totals for each class are the sums across the rows (axis=1)
    actual_totals = np.sum(confusion, axis=1)

    # Sensitivity = TP / (TP + FN) -> Which is identical to TP / Actual Totals
    return true_positives / actual_totals
