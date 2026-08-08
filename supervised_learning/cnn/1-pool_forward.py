#!/usr/bin/env python3
"""Pooling Forward Prop module."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a
    neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the
            kernel for the pooling
        stride: tuple of (sh, sw) containing the strides for the pooling
        mode: string, either 'max' or 'avg'

    Returns:
        The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    A = np.zeros((m, output_h, output_w, c_prev))

    if mode == 'max':
        op = np.max
    else:
        op = np.average

    for i in range(output_h):
        for j in range(output_w):
            y = i * sh
            x = j * sw
            A[:, i, j, :] = op(
                A_prev[:, y:y + kh, x:x + kw, :],
                axis=(1, 2)
            )

    return A
