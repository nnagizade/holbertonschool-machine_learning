#!/usr/bin/env python3
"""RNN Cell Module."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple Recurrent Neural Network (RNN)."""

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell attributes.

        Parameters:
            i (int): Dimensionality of the data input
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
            h_prev (np.ndarray): Shape (m, h) containing previous hidden state
            x_t (np.ndarray): Shape (m, i) containing input data for cell

        Returns:
            h_next (np.ndarray): Next hidden state
            y (np.ndarray): Output of the cell after softmax activation
        """
        # Concatenate previous hidden state and current input along axis 1
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Compute output linear transformation and apply Softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
