#!/usr/bin/env python3
"""
Module to update variables using Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    Args:
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var
        s: previous second moment of var
        t: time step used for bias correction

    Returns:
        The updated variable, the new first moment, and the new second moment
    """
    # Update biased first moment estimate
    v_new = beta1 * v + (1 - beta1) * grad

    # Update biased second raw moment estimate
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first moment estimate
    v_corrected = v_new / (1 - (beta1 ** t))

    # Compute bias-corrected second raw moment estimate
    s_corrected = s_new / (1 - (beta2 ** t))

    # Update variable
    var_updated = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)

    return var_updated, v_new, s_new
