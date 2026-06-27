#!/usr/bin/env python3
"""
Module to create a confusion matrix using numpy
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): a one-hot encoded numpy.ndarray of shape
                                (m, classes) containing the correct labels.
        logits (numpy.ndarray): a one-hot encoded numpy.ndarray of shape
                                (m, classes) containing the predicted labels.

    Returns:
        numpy.ndarray: a confusion matrix of shape (classes, classes) with row
                       indices representing the correct labels and column
                       indices representing the predicted labels.
    """
    # The dot product of labels.T (classes, m) and logits (m, classes)
    # automatically counts the joint occurrences of true class i and pred class j
    return np.dot(labels.T, logits)
