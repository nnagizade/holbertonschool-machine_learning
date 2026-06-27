#!/usr/bin/env python3
"""Calculates sensitivity for each class"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    return np.diag(confusion) / np.sum(confusion, axis=1)
