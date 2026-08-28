#!/usr/bin/env python3
"""Calculates the positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: integer, maximum sequence length
        dm: model depth

    Returns:
        a numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.power(10000, (2 * (np.arange(dm) // 2)) / dm)

    angles = position / div_term

    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
