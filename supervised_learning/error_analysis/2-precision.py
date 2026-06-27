#!/usr/bin/env python3
"""
Module to calculate the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): a confusion matrix of shape (classes, classes)
                                   where row indices represent correct labels
                                   and column indices represent predicted labels

    Returns:
        numpy.ndarray: a numpy.ndarray of shape (classes,) containing the
                       precision of each class
    """
    # True Positives lie along the main diagonal
    true_positives = np.diagonal(confusion)

    # Predicted totals for each class are the column sums (axis=0)
    predicted_totals = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP) -> Which is identical to TP / Predicted Totals
    return true_positives / predicted_totals
