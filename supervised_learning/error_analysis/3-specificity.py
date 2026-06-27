#!/usr/bin/env python3
"""
Module to calculate the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): a confusion matrix of shape (classes, classes)
                                   where row indices represent correct labels
                                   and column indices represent predicted labels

    Returns:
        numpy.ndarray: a numpy.ndarray of shape (classes,) containing the
                       specificity of each class
    """
    # Total number of samples in the confusion matrix
    total_samples = np.sum(confusion)

    # True Positives (TP) along the main diagonal
    true_positives = np.diagonal(confusion)

    # Row sums (Actual class totals)
    actual_totals = np.sum(confusion, axis=1)

    # Column sums (Predicted class totals)
    predicted_totals = np.sum(confusion, axis=0)

    # False Positives (FP) = Column sum - TP
    false_positives = predicted_totals - true_positives

    # False Negatives (FN) = Row sum - TP
    false_negatives = actual_totals - true_positives

    # True Negatives (TN) = Total - TP - FP - FN
    true_negatives = total_samples - true_positives - false_positives - false_negatives

    # Specificity = TN / (TN + FP)
    return true_negatives / (true_negatives + false_positives)
