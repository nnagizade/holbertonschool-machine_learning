#!/usr/bin/env python3
"""Strided Convolution module."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w) containing multiple
            grayscale images
        kernel: numpy.ndarray of shape (kh, kw) containing the kernel
            for the convolution
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(images,
                     ((0, 0), (ph, ph), (pw, pw)),
                     mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            y = i * sh
            x = j * sw
            output[:, i, j] = np.sum(
                padded[:, y:y + kh, x:x + kw] * kernel,
                axis=(1, 2)
            )

    return output
