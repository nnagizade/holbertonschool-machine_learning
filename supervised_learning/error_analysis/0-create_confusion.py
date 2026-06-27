#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    return np.dot(labels.T, logits)
