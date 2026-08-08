#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization module."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient

    descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A{}'.format(i - 1)]
        W_key = 'W{}'.format(i)
        b_key = 'b{}'.format(i)

        # Calculate gradients with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) + (lambtha * weights[W_key])) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            # Backpropagate through tanh activation derivative: 1 - A^2
            dZ = np.matmul(weights[W_key].T, dZ) * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db
