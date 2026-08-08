#!/usr/bin/env python3
"""Convolutional Forward Prop module."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a
    neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the
            biases applied to the convolution
        activation: an activation function applied to the convolution
        padding: string, either "same" or "valid"
        stride: tuple of (sh, sw) containing the strides

    Returns:
        The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    padded = np.pad(A_prev,
                     ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')

    output_h = (h_prev + 2 * ph - kh) // sh + 1
    output_w = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            y = i * sh
            x = j * sw
            box = padded[:, y:y + kh, x:x + kw, :]
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    box * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return activation(Z + b)
