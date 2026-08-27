#!/usr/bin/env python3
"""LSTM Cell module."""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initializes the LSTM cell weights and biases.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            c_prev: numpy.ndarray of shape (m, h) with previous cell state
            x_t: numpy.ndarray of shape (m, i) with input data for the cell

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        c_next = f * c_prev + u * c_tilde

        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))
        h_next = o * np.tanh(c_next)

        y_lin = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_lin - np.max(y_lin, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
