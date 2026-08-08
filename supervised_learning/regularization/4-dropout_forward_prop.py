#!/usr/bin/env python3
"""Forward Propagation with Dropout module."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
        weights: dictionary of the weights and biases of the neural network
        L: the number of layers in the network
        keep_prob: the probability that a node will be kept

    Returns:
        A dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W{}'.format(i)]
        b = weights['b{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = A * D
            A = A / keep_prob
            cache['D{}'.format(i)] = D

        cache['A{}'.format(i)] = A

    return cache
