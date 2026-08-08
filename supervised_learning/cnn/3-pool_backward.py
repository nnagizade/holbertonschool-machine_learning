#!/usr/bin/env python3
"""Pooling Back Prop module."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a
    neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output of the
            pooling layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing
            the output of the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the
            kernel for the pooling
        stride: tuple of (sh, sw) containing the strides for the pooling
        mode: string, either 'max' or 'avg'

    Returns:
        The partial derivatives with respect to the previous layer
        (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    y = h * sh
                    x = w * sw

                    if mode == 'max':
                        a_slice = A_prev[i, y:y + kh, x:x + kw, ch]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, y:y + kh, x:x + kw, ch] += (
                            mask * dA[i, h, w, ch]
                        )
                    elif mode == 'avg':
                        da = dA[i, h, w, ch]
                        average = da / (kh * kw)
                        dA_prev[i, y:y + kh, x:x + kw, ch] += (
                            np.ones((kh, kw)) * average
                        )

    return dA_prev
