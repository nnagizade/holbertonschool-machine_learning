#!/usr/bin/env python3
"""Convolutional Back Prop module."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a
    neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated
            output of the convolutional layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the
            biases applied to the convolution
        padding: string, either "same" or "valid"
        stride: tuple of (sh, sw) containing the strides

    Returns:
        The partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
        ph = max(int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2)), 0)
        pw = max(int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2)), 0)
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev,
                         ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant')

    dA_prev_pad = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    y = h * sh
                    x = w * sw

                    a_slice = a_prev_pad[y:y + kh, x:x + kw, :]

                    da_prev_pad[y:y + kh, x:x + kw, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == 'same':
            dA_prev_pad_i = da_prev_pad[ph:ph + h_prev, pw:pw + w_prev, :]
        else:
            dA_prev_pad_i = da_prev_pad

        dA_prev_pad[i] = da_prev_pad
        if i == 0:
            dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
        dA_prev[i] = dA_prev_pad_i

    return dA_prev, dW, db
