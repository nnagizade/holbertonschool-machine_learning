#!/usr/bin/env python3
"""
Module to calculate the F1 score for each class in a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): a confusion matrix of shape (classes, classes)
                                   where row indices represent correct labels
                                   and column indices represent predicted labels

    Returns:
        numpy.ndarray: a numpy.ndarray of shape (classes,) containing the
                       F1 score of each class
    """
    # Calculate sensitivity (recall) and precision per class
    se = sensitivity(confusion)
    pr = precision(confusion)

    # F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    # Using np.where to elegantly handle any potential division by zero
    f1 = 2 * (pr * se) / (pr + se)
    f1 = np.where(np.isnan(f1), 0, f1)

    return f1
