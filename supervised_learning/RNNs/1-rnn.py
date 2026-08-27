#!/usr/bin/env python3
"""Simple RNN Forward Propagation module."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN across multiple time steps.

    Parameters:
        rnn_cell (RNNCell): Instance of RNNCell used for forward propagation
        X (np.ndarray): Data of shape (t, m, i) where:
            t is maximum number of time steps
            m is batch size
            i is dimensionality of input data
        h_0 (np.ndarray): Initial hidden state of shape (m, h) where:
            h is dimensionality of hidden state

    Returns:
        H (np.ndarray): Shape (t + 1, m, h) containing all hidden states
        Y (np.ndarray): Shape (t, m, o) containing all outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
