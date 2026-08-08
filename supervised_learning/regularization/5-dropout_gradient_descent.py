#!/usr/bin/env python3
"""Gradient Descent with Dropout module."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network

    Returns:
        None. Updates weights in place.
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    dZ = cache['A{}'.format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A{}'.format(i - 1)]
        W = weights_copy['W{}'.format(i)]
        b = weights_copy['b{}'.format(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = dA_prev * cache['D{}'.format(i - 1)]
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W{}'.format(i)] = W - alpha * dW
        weights['b{}'.format(i)] = b - alpha * db
