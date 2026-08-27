#!/usr/bin/env python3
"""Gated Recurrent Unit (GRU) Cell module."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell."""

    def __init__(self, i, h, o):
        """
        Initializes the GRU Cell weights and biases.

        Parameters:
            i (int): Dimensionality of input data
            h (int): Dimensionality of hidden state
            o (int): Dimensionality of output data
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
            h_prev (np.ndarray): Shape (m, h) containing previous hidden state
            x_t (np.ndarray): Shape (m, i) containing input data for cell

        Returns:
            h_next (np.ndarray): Shape (m, h) containing next hidden state
            y (np.ndarray): Shape (m, o) containing output probabilities
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update Gate (z)
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))

        # Reset Gate (r)
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        # Intermediate Hidden State (h_tilde)
        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next Hidden State (h_next)
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output Prediction (y) using Softmax
        y_lin = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_lin - np.max(y_lin, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
