#!/usr/bin/env python3
"""Deep RNN forward propagation"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
    - rnn_cells: list of RNNCell instances of length l
    - X: numpy.ndarray of shape (t, m, i) containing input data
    - h_0: numpy.ndarray of shape (l, m, h) containing initial hidden state

    Returns:
    - H: numpy.ndarray of shape (t + 1, l, m, h) containing all hidden states
    - Y: numpy.ndarray of shape (t, m, o) containing all outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            H[step + 1, layer] = h_next
            x_t = h_next

        if step == 0:
            Y = np.zeros((t, m, y.shape[1]))
        Y[step] = y

    return H, Y
