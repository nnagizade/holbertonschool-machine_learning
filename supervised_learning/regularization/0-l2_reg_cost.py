#!/usr/bin/env python3
"""Module to calculate L2 regularization cost."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of weights and biases (numpy.ndarrays)
        L: number of layers in the neural network
        m: number of data points used

    Returns:
        The total cost of the network accounting for L2 regularization
    """
    l2_term = 0
    for i in range(1, L + 1):
        l2_term += np.sum(np.square(weights['W{}'.format(i)]))

    return cost + (lambtha / (2 * m)) * l2_term
