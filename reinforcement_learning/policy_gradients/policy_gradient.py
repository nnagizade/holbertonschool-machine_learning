#!/usr/bin/env python3
"""Simple Policy function module."""
import numpy as np


def policy(matrix, weight):
    """
    Compute the policy with a weight of a matrix.

    Args:
        matrix: numpy.ndarray representing the state
        weight: numpy.ndarray representing the weight

    Returns:
        numpy.ndarray of action probabilities (the policy)
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient based on a state and weight matrix.

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Returns:
        the action and the gradient (in this order)
    """
    P = policy(state, weight)
    probs = P[0] / P[0].sum()
    action = np.random.choice(len(probs), p=probs)

    s = P.reshape(-1, 1)
    softmax_grad = np.diagflat(s) - np.dot(s, s.T)
    dsoftmax = softmax_grad[action, :]
    dlog = dsoftmax / P[0, action]
    gradient = state.T.dot(dlog[None, :])

    return action, gradient
