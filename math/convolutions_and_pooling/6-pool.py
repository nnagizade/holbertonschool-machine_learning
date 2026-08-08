#!/usr/bin/env python3
"""Pooling module."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images: numpy.ndarray of shape (m, h, w, c) containing multiple
            images
        kernel_shape: tuple of (kh, kw) containing the kernel shape for
            the pooling
        stride: tuple of (sh, sw)
        mode: indicates the type of pooling ('max' or 'avg')

    Returns:
        A numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, c))

    if mode == 'max':
        op = np.max
    else:
        op = np.average

    for i in range(output_h):
        for j in range(output_w):
            y = i * sh
            x = j * sw
            output[:, i, j, :] = op(
                images[:, y:y + kh, x:x + kw, :],
                axis=(1, 2)
            )

    return output
