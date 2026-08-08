#!/usr/bin/env python3
"""Multiple Kernels module."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray of shape (m, h, w, c) containing multiple
            images
        kernels: numpy.ndarray of shape (kh, kw, c, nc) containing the
            kernels for the convolution
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(images,
                     ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                y = i * sh
                x = j * sw
                output[:, i, j, k] = np.sum(
                    padded[:, y:y + kh, x:x + kw, :] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
